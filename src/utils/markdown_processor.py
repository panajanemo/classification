import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from src.utils.text_processor import TextProcessor

logger = logging.getLogger(__name__)


@dataclass
class MarkdownBlock:
    """Markdown块结构"""
    type: str  # paragraph, heading, code, list, quote, etc.
    content: str
    level: Optional[int] = None  # 标题级别
    metadata: Optional[Dict[str, Any]] = None
    line_start: int = 0
    line_end: int = 0


class MarkdownProcessor:
    """Markdown文档处理器"""
    
    def __init__(self, text_processor: Optional[TextProcessor] = None):
        """
        初始化Markdown处理器
        
        Args:
            text_processor: 文本处理器实例
        """
        self.text_processor = text_processor or TextProcessor()
        
        # Markdown语法规则
        self.patterns = {
            'heading': r'^(#{1,6})\s+(.+)$',
            'code_block': r'^```(\w*)\n(.*?)^```$',
            'inline_code': r'`([^`]+)`',
            'bold': r'\*\*([^*]+)\*\*',
            'italic': r'\*([^*]+)\*',
            'link': r'\[([^\]]+)\]\(([^)]+)\)',
            'image': r'!\[([^\]]*)\]\(([^)]+)\)',
            'list_item': r'^(\s*)([-*+]|\d+\.)\s+(.+)$',
            'quote': r'^>\s*(.+)$',
            'horizontal_rule': r'^[-*_]{3,}$',
            'table_row': r'^\|(.+)\|$'
        }
    
    def parse_markdown(self, markdown_content: str) -> List[MarkdownBlock]:
        """
        解析Markdown内容为结构化块
        
        Args:
            markdown_content: Markdown文本内容
            
        Returns:
            结构化的Markdown块列表
        """
        if not markdown_content:
            return []
        
        lines = markdown_content.split('\n')
        blocks = []
        current_block = None
        in_code_block = False
        code_block_content = []
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # 处理代码块
            if line_stripped.startswith('```'):
                if not in_code_block:
                    # 开始代码块
                    in_code_block = True
                    code_block_content = []
                    language = line_stripped[3:].strip()
                    current_block = MarkdownBlock(
                        type='code_block',
                        content='',
                        metadata={'language': language},
                        line_start=i
                    )
                else:
                    # 结束代码块
                    in_code_block = False
                    if current_block:
                        current_block.content = '\n'.join(code_block_content)
                        current_block.line_end = i
                        blocks.append(current_block)
                        current_block = None
                continue
            
            if in_code_block:
                code_block_content.append(line)
                continue
            
            # 空行处理
            if not line_stripped:
                if current_block and current_block.type == 'paragraph':
                    current_block.line_end = i - 1
                    blocks.append(current_block)
                    current_block = None
                continue
            
            # 标题
            heading_match = re.match(self.patterns['heading'], line_stripped)
            if heading_match:
                if current_block:
                    current_block.line_end = i - 1
                    blocks.append(current_block)
                
                level = len(heading_match.group(1))
                content = heading_match.group(2)
                blocks.append(MarkdownBlock(
                    type='heading',
                    content=content,
                    level=level,
                    line_start=i,
                    line_end=i
                ))
                current_block = None
                continue
            
            # 引用
            quote_match = re.match(self.patterns['quote'], line_stripped)
            if quote_match:
                if current_block and current_block.type != 'quote':
                    current_block.line_end = i - 1
                    blocks.append(current_block)
                    current_block = None
                
                if not current_block:
                    current_block = MarkdownBlock(
                        type='quote',
                        content=quote_match.group(1),
                        line_start=i
                    )
                else:
                    current_block.content += ' ' + quote_match.group(1)
                continue
            
            # 列表项
            list_match = re.match(self.patterns['list_item'], line_stripped)
            if list_match:
                if current_block and current_block.type != 'list':
                    current_block.line_end = i - 1
                    blocks.append(current_block)
                    current_block = None
                
                indent = len(list_match.group(1))
                marker = list_match.group(2)
                content = list_match.group(3)
                
                if not current_block:
                    current_block = MarkdownBlock(
                        type='list',
                        content=f"{' ' * indent}{marker} {content}",
                        metadata={'items': [content]},
                        line_start=i
                    )
                else:
                    current_block.content += f"\n{' ' * indent}{marker} {content}"
                    current_block.metadata['items'].append(content)
                continue
            
            # 水平线
            if re.match(self.patterns['horizontal_rule'], line_stripped):
                if current_block:
                    current_block.line_end = i - 1
                    blocks.append(current_block)
                
                blocks.append(MarkdownBlock(
                    type='horizontal_rule',
                    content=line_stripped,
                    line_start=i,
                    line_end=i
                ))
                current_block = None
                continue
            
            # 表格行
            if re.match(self.patterns['table_row'], line_stripped):
                if current_block and current_block.type != 'table':
                    current_block.line_end = i - 1
                    blocks.append(current_block)
                    current_block = None
                
                if not current_block:
                    current_block = MarkdownBlock(
                        type='table',
                        content=line_stripped,
                        metadata={'rows': [line_stripped]},
                        line_start=i
                    )
                else:
                    current_block.content += '\n' + line_stripped
                    current_block.metadata['rows'].append(line_stripped)
                continue
            
            # 普通段落
            if current_block and current_block.type == 'paragraph':
                current_block.content += ' ' + line_stripped
            else:
                if current_block:
                    current_block.line_end = i - 1
                    blocks.append(current_block)
                
                current_block = MarkdownBlock(
                    type='paragraph',
                    content=line_stripped,
                    line_start=i
                )
        
        # 处理最后一个块
        if current_block:
            current_block.line_end = len(lines) - 1
            blocks.append(current_block)
        
        logger.info(f"解析Markdown完成，共{len(blocks)}个块")
        return blocks
    
    def segment_markdown_blocks(self, blocks: List[MarkdownBlock], 
                              threshold: float = 0.5) -> List[MarkdownBlock]:
        """
        对Markdown块进行语义分段
        
        Args:
            blocks: 解析后的Markdown块
            threshold: 分段阈值
            
        Returns:
            分段后的Markdown块
        """
        segmented_blocks = []
        
        for block in blocks:
            # 只对段落类型进行语义分段
            if block.type == 'paragraph' and len(block.content) > 200:
                try:
                    # 使用语义分段器处理长段落
                    from src.core.semantic_segmenter import SemanticSegmenter
                    from src.core.bert_model import BERTModel
                    
                    # 创建临时分段器（或使用全局实例）
                    bert_model = BERTModel()
                    segmenter = SemanticSegmenter(bert_model, self.text_processor)
                    segmenter.set_threshold(threshold)
                    
                    # 执行分段
                    result = segmenter.segment_text(block.content)
                    
                    if "paragraphs" in result and len(result["paragraphs"]) > 1:
                        # 将长段落分割为多个子段落
                        for i, paragraph in enumerate(result["paragraphs"]):
                            new_block = MarkdownBlock(
                                type='paragraph',
                                content=paragraph.strip(),
                                line_start=block.line_start,
                                line_end=block.line_end
                            )
                            segmented_blocks.append(new_block)
                        
                        logger.debug(f"段落分段: 1 -> {len(result['paragraphs'])}段")
                    else:
                        segmented_blocks.append(block)
                        
                except Exception as e:
                    logger.warning(f"段落分段失败: {e}, 保持原始段落")
                    segmented_blocks.append(block)
            else:
                # 其他类型的块保持不变
                segmented_blocks.append(block)
        
        return segmented_blocks
    
    def reconstruct_markdown(self, blocks: List[MarkdownBlock]) -> str:
        """
        从结构化块重构Markdown文本
        
        Args:
            blocks: 结构化的Markdown块
            
        Returns:
            重构后的Markdown文本
        """
        lines = []
        
        for i, block in enumerate(blocks):
            if block.type == 'heading':
                prefix = '#' * (block.level or 1)
                lines.append(f"{prefix} {block.content}")
            
            elif block.type == 'code_block':
                language = block.metadata.get('language', '') if block.metadata else ''
                lines.append(f"```{language}")
                lines.append(block.content)
                lines.append("```")
            
            elif block.type == 'quote':
                # 处理引用块
                for line in block.content.split('\n'):
                    lines.append(f"> {line}")
            
            elif block.type == 'list':
                lines.append(block.content)
            
            elif block.type == 'table':
                lines.append(block.content)
            
            elif block.type == 'horizontal_rule':
                lines.append(block.content)
            
            elif block.type == 'paragraph':
                lines.append(block.content)
            
            # 在块之间添加空行（除了最后一个块）
            if i < len(blocks) - 1:
                next_block = blocks[i + 1]
                # 某些情况下不需要额外空行
                if (block.type in ['list', 'quote', 'table'] and 
                    next_block.type == block.type):
                    continue
                lines.append("")
        
        return '\n'.join(lines)
    
    def extract_text_content(self, blocks: List[MarkdownBlock]) -> str:
        """
        从Markdown块中提取纯文本内容
        
        Args:
            blocks: Markdown块列表
            
        Returns:
            提取的纯文本
        """
        text_parts = []
        
        for block in blocks:
            if block.type in ['paragraph', 'heading', 'quote']:
                # 移除Markdown语法
                content = block.content
                # 移除链接语法
                content = re.sub(self.patterns['link'], r'\1', content)
                # 移除图片语法
                content = re.sub(self.patterns['image'], r'\1', content)
                # 移除粗体和斜体
                content = re.sub(self.patterns['bold'], r'\1', content)
                content = re.sub(self.patterns['italic'], r'\1', content)
                # 移除行内代码
                content = re.sub(self.patterns['inline_code'], r'\1', content)
                
                text_parts.append(content.strip())
            
            elif block.type == 'list' and block.metadata:
                # 提取列表项内容
                for item in block.metadata.get('items', []):
                    text_parts.append(item.strip())
        
        return '\n\n'.join(text_parts)
    
    def get_statistics(self, blocks: List[MarkdownBlock]) -> Dict[str, Any]:
        """
        获取Markdown文档统计信息
        
        Args:
            blocks: Markdown块列表
            
        Returns:
            统计信息字典
        """
        stats = {
            'total_blocks': len(blocks),
            'block_types': {},
            'headings': {'count': 0, 'levels': {}},
            'paragraphs': {'count': 0, 'total_chars': 0},
            'code_blocks': 0,
            'lists': 0,
            'tables': 0
        }
        
        for block in blocks:
            # 统计块类型
            stats['block_types'][block.type] = stats['block_types'].get(block.type, 0) + 1
            
            if block.type == 'heading':
                stats['headings']['count'] += 1
                level = block.level or 1
                stats['headings']['levels'][level] = stats['headings']['levels'].get(level, 0) + 1
            
            elif block.type == 'paragraph':
                stats['paragraphs']['count'] += 1
                stats['paragraphs']['total_chars'] += len(block.content)
            
            elif block.type == 'code_block':
                stats['code_blocks'] += 1
            
            elif block.type == 'list':
                stats['lists'] += 1
            
            elif block.type == 'table':
                stats['tables'] += 1
        
        # 计算平均段落长度
        if stats['paragraphs']['count'] > 0:
            stats['paragraphs']['avg_length'] = stats['paragraphs']['total_chars'] / stats['paragraphs']['count']
        else:
            stats['paragraphs']['avg_length'] = 0
        
        return stats
    
    def process_markdown_document(self, markdown_content: str, 
                                threshold: float = 0.5) -> Dict[str, Any]:
        """
        完整处理Markdown文档
        
        Args:
            markdown_content: 原始Markdown内容
            threshold: 分段阈值
            
        Returns:
            处理结果字典
        """
        try:
            logger.info("开始处理Markdown文档")
            
            # 1. 解析Markdown结构
            original_blocks = self.parse_markdown(markdown_content)
            original_stats = self.get_statistics(original_blocks)
            
            # 2. 语义分段处理
            segmented_blocks = self.segment_markdown_blocks(original_blocks, threshold)
            segmented_stats = self.get_statistics(segmented_blocks)
            
            # 3. 重构Markdown
            processed_markdown = self.reconstruct_markdown(segmented_blocks)
            
            # 4. 提取文本内容
            text_content = self.extract_text_content(segmented_blocks)
            
            result = {
                'success': True,
                'original_stats': original_stats,
                'processed_stats': segmented_stats,
                'processed_markdown': processed_markdown,
                'text_content': text_content,
                'blocks_count': {
                    'original': len(original_blocks),
                    'processed': len(segmented_blocks)
                },
                'processing_info': {
                    'threshold': threshold,
                    'segmentation_applied': len(segmented_blocks) > len(original_blocks)
                }
            }
            
            logger.info(f"Markdown处理完成: {len(original_blocks)} -> {len(segmented_blocks)}块")
            return result
            
        except Exception as e:
            logger.error(f"Markdown处理失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'original_markdown': markdown_content
            }