import logging
import asyncio
import uuid
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import time

from src.utils.markdown_processor import MarkdownProcessor, MarkdownBlock
from src.core.semantic_segmenter import SemanticSegmenter

logger = logging.getLogger(__name__)


@dataclass
class ProcessingTask:
    """处理任务"""
    task_id: str
    filename: str
    content: str
    threshold: float
    chunk_size: int
    preserve_structure: bool
    status: str = "pending"
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None


class ChunkProcessor:
    """分块处理引擎"""
    
    def __init__(self, segmenter: SemanticSegmenter):
        """
        初始化分块处理器
        
        Args:
            segmenter: 语义分段器实例
        """
        self.segmenter = segmenter
        self.markdown_processor = MarkdownProcessor(segmenter.text_processor)
        self.tasks: Dict[str, ProcessingTask] = {}
        self.executor = ThreadPoolExecutor(max_workers=3)  # 限制并发数
        
        logger.info("分块处理引擎初始化完成")
    
    def create_task(self, filename: str, content: str, threshold: float = 0.5,
                   chunk_size: int = 5000, preserve_structure: bool = True) -> str:
        """
        创建处理任务
        
        Args:
            filename: 文件名
            content: 文件内容
            threshold: 分段阈值
            chunk_size: 分块大小
            preserve_structure: 是否保持结构
            
        Returns:
            任务ID
        """
        task_id = f"md_proc_{uuid.uuid4().hex[:8]}"
        
        task = ProcessingTask(
            task_id=task_id,
            filename=filename,
            content=content,
            threshold=threshold,
            chunk_size=chunk_size,
            preserve_structure=preserve_structure
        )
        
        self.tasks[task_id] = task
        logger.info(f"创建处理任务: {task_id}, 文件: {filename}")
        
        return task_id
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        获取任务状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务状态信息
        """
        task = self.tasks.get(task_id)
        if not task:
            return None
        
        estimated_time = None
        if task.status == "processing" and task.start_time:
            elapsed = time.time() - task.start_time
            if task.progress > 0:
                estimated_total = elapsed / (task.progress / 100)
                estimated_time = max(0, estimated_total - elapsed)
        
        return {
            "task_id": task_id,
            "status": task.status,
            "progress": task.progress,
            "message": self._get_status_message(task),
            "estimated_time": estimated_time,
            "result_available": task.result is not None
        }
    
    def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        获取任务结果
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务结果
        """
        task = self.tasks.get(task_id)
        if not task or task.status != "completed":
            return None
        
        return task.result
    
    def _get_status_message(self, task: ProcessingTask) -> str:
        """获取状态消息"""
        if task.status == "pending":
            return "等待处理"
        elif task.status == "processing":
            if task.progress < 10:
                return "正在解析Markdown结构..."
            elif task.progress < 50:
                return "正在进行语义分段..."
            elif task.progress < 80:
                return "正在重构文档..."
            else:
                return "正在完成处理..."
        elif task.status == "completed":
            return "处理完成"
        elif task.status == "failed":
            return f"处理失败: {task.error}"
        else:
            return "未知状态"
    
    def _split_into_chunks(self, blocks: List[MarkdownBlock], 
                          chunk_size: int) -> List[List[MarkdownBlock]]:
        """
        将Markdown块分割成处理块
        
        Args:
            blocks: Markdown块列表
            chunk_size: 目标块大小
            
        Returns:
            分块后的列表
        """
        chunks = []
        current_chunk = []
        current_size = 0
        
        for block in blocks:
            block_size = len(block.content)
            
            # 如果单个块就超过了chunk_size，单独处理
            if block_size > chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = [block]
                current_size = block_size
            elif current_size + block_size > chunk_size and current_chunk:
                # 当前块会超出大小限制，先保存当前chunk
                chunks.append(current_chunk)
                current_chunk = [block]
                current_size = block_size
            else:
                # 添加到当前chunk
                current_chunk.append(block)
                current_size += block_size
        
        # 添加最后一个chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        logger.debug(f"分块完成: {len(blocks)}块 -> {len(chunks)}个处理块")
        return chunks
    
    def _process_chunk(self, chunk: List[MarkdownBlock], 
                      threshold: float) -> List[MarkdownBlock]:
        """
        处理单个块
        
        Args:
            chunk: 要处理的Markdown块
            threshold: 分段阈值
            
        Returns:
            处理后的块
        """
        try:
            return self.markdown_processor.segment_markdown_blocks(chunk, threshold)
        except Exception as e:
            logger.error(f"块处理失败: {e}")
            return chunk  # 返回原始块
    
    def _update_progress(self, task: ProcessingTask, progress: float, 
                        force_update: bool = False):
        """更新任务进度"""
        if force_update or progress - task.progress >= 5.0:  # 至少5%的变化才更新
            task.progress = min(100.0, progress)
            logger.debug(f"任务 {task.task_id} 进度: {task.progress:.1f}%")
    
    async def process_markdown_async(self, task_id: str) -> Dict[str, Any]:
        """
        异步处理Markdown文档
        
        Args:
            task_id: 任务ID
            
        Returns:
            处理结果
        """
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"任务不存在: {task_id}")
        
        try:
            task.status = "processing"
            task.start_time = time.time()
            self._update_progress(task, 0.0, force_update=True)
            
            logger.info(f"开始处理任务: {task_id}")
            
            # 1. 解析Markdown结构 (0-10%)
            original_blocks = self.markdown_processor.parse_markdown(task.content)
            original_stats = self.markdown_processor.get_statistics(original_blocks)
            self._update_progress(task, 10.0)
            
            # 2. 判断是否需要分块处理
            total_content_size = sum(len(block.content) for block in original_blocks)
            
            if total_content_size <= task.chunk_size:
                # 小文件直接处理 (10-90%)
                logger.info(f"小文件直接处理: {total_content_size} 字符")
                self._update_progress(task, 30.0)
                
                segmented_blocks = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self._process_chunk,
                    original_blocks,
                    task.threshold
                )
                self._update_progress(task, 70.0)
                
            else:
                # 大文件分块处理 (10-80%)
                logger.info(f"大文件分块处理: {total_content_size} 字符")
                chunks = self._split_into_chunks(original_blocks, task.chunk_size)
                self._update_progress(task, 20.0)
                
                # 并行处理各个块
                processed_chunks = []
                chunk_progress_step = 50.0 / len(chunks)  # 20% -> 70%
                
                for i, chunk in enumerate(chunks):
                    processed_chunk = await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        self._process_chunk,
                        chunk,
                        task.threshold
                    )
                    processed_chunks.append(processed_chunk)
                    
                    progress = 20.0 + (i + 1) * chunk_progress_step
                    self._update_progress(task, progress)
                
                # 合并处理结果
                segmented_blocks = []
                for chunk in processed_chunks:
                    segmented_blocks.extend(chunk)
                
                self._update_progress(task, 70.0)
            
            # 3. 重构Markdown (70-90%)
            processed_markdown = self.markdown_processor.reconstruct_markdown(segmented_blocks)
            self._update_progress(task, 80.0)
            
            # 4. 生成统计信息 (90-100%)
            processed_stats = self.markdown_processor.get_statistics(segmented_blocks)
            text_content = self.markdown_processor.extract_text_content(segmented_blocks)
            
            task.end_time = time.time()
            processing_time = task.end_time - task.start_time
            
            # 构建结果
            result = {
                'success': True,
                'filename': task.filename,
                'file_size': len(task.content.encode('utf-8')),
                'processed_markdown': processed_markdown,
                'original_stats': original_stats,
                'processed_stats': processed_stats,
                'blocks_count': {
                    'original': len(original_blocks),
                    'processed': len(segmented_blocks)
                },
                'processing_info': {
                    'threshold': task.threshold,
                    'chunk_size': task.chunk_size,
                    'preserve_structure': task.preserve_structure,
                    'segmentation_applied': len(segmented_blocks) > len(original_blocks),
                    'chunks_used': len(chunks) if total_content_size > task.chunk_size else 1
                },
                'processing_time': processing_time
            }
            
            task.result = result
            task.status = "completed"
            self._update_progress(task, 100.0, force_update=True)
            
            logger.info(f"任务完成: {task_id}, 耗时: {processing_time:.2f}秒")
            return result
            
        except Exception as e:
            logger.error(f"任务处理失败: {task_id}, 错误: {e}")
            task.status = "failed"
            task.error = str(e)
            task.end_time = time.time()
            
            return {
                'success': False,
                'error': str(e),
                'filename': task.filename
            }
    
    def process_markdown_sync(self, filename: str, content: str, 
                            threshold: float = 0.5, chunk_size: int = 5000,
                            preserve_structure: bool = True) -> Dict[str, Any]:
        """
        同步处理Markdown文档（简化版）
        
        Args:
            filename: 文件名
            content: 文件内容
            threshold: 分段阈值
            chunk_size: 分块大小
            preserve_structure: 是否保持结构
            
        Returns:
            处理结果
        """
        try:
            start_time = time.time()
            logger.info(f"开始同步处理: {filename}")
            
            # 直接使用markdown处理器
            result = self.markdown_processor.process_markdown_document(content, threshold)
            
            if result['success']:
                result.update({
                    'filename': filename,
                    'file_size': len(content.encode('utf-8')),
                    'processing_time': time.time() - start_time
                })
            
            logger.info(f"同步处理完成: {filename}")
            return result
            
        except Exception as e:
            logger.error(f"同步处理失败: {filename}, 错误: {e}")
            return {
                'success': False,
                'error': str(e),
                'filename': filename
            }
    
    def cleanup_completed_tasks(self, max_age_hours: int = 24):
        """
        清理完成的任务
        
        Args:
            max_age_hours: 最大保留时间（小时）
        """
        current_time = time.time()
        cutoff_time = current_time - (max_age_hours * 3600)
        
        to_remove = []
        for task_id, task in self.tasks.items():
            if (task.status in ["completed", "failed"] and 
                task.end_time and task.end_time < cutoff_time):
                to_remove.append(task_id)
        
        for task_id in to_remove:
            del self.tasks[task_id]
            logger.debug(f"清理任务: {task_id}")
        
        if to_remove:
            logger.info(f"清理了 {len(to_remove)} 个过期任务")
    
    def get_all_tasks_status(self) -> List[Dict[str, Any]]:
        """获取所有任务状态"""
        return [self.get_task_status(task_id) for task_id in self.tasks.keys()]
    
    def cancel_task(self, task_id: str) -> bool:
        """
        取消任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            是否成功取消
        """
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        if task.status == "pending":
            task.status = "cancelled"
            logger.info(f"任务已取消: {task_id}")
            return True
        
        return False