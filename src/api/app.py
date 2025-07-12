import time
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.api.models import (
    SegmentationRequest, SegmentationResponse, ErrorResponse,
    HealthResponse, ConfigRequest, ConfigResponse, QualityMetrics,
    MarkdownUploadRequest, MarkdownProcessingResponse, MarkdownProcessingStats,
    FileUploadErrorResponse, ProcessingStatusResponse
)
from src.core.semantic_segmenter import SemanticSegmenter
from src.core.bert_model import BERTModel
from src.utils.text_processor import TextProcessor
from src.core.chunk_processor import ChunkProcessor
from config.settings import settings

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量
segmenter = None
chunk_processor = None
startup_time = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global segmenter, chunk_processor, startup_time
    
    # 启动时初始化
    logger.info("正在启动语义分段服务...")
    startup_time = time.time()
    
    try:
        # 初始化组件
        bert_model = BERTModel()
        text_processor = TextProcessor()
        segmenter = SemanticSegmenter(bert_model, text_processor)
        
        # 初始化分块处理器
        chunk_processor = ChunkProcessor(segmenter)
        
        # 预加载模型
        logger.info("正在预加载BERT模型...")
        segmenter.bert_model.load_model()
        
        logger.info("语义分段服务启动成功")
        yield
        
    except Exception as e:
        logger.error(f"服务启动失败: {e}")
        raise
    finally:
        # 关闭时清理
        logger.info("正在关闭语义分段服务...")
        # 清理分块处理器
        if chunk_processor:
            chunk_processor.cleanup_completed_tasks()


# 创建FastAPI应用
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="基于BERT的语义分段服务API",
    lifespan=lifespan
)

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_segmenter() -> SemanticSegmenter:
    """获取分段器实例"""
    if segmenter is None:
        raise HTTPException(status_code=503, detail="服务尚未初始化完成")
    return segmenter


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理器"""
    logger.error(f"未处理的异常: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="内部服务器错误",
            error_type=type(exc).__name__,
            details={"message": str(exc)}
        ).dict()
    )


@app.get("/", response_class=HTMLResponse)
async def root():
    """根路径 - 重定向到Web界面"""
    try:
        with open("templates/upload.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        return {
            "message": "欢迎使用BERT语义分段服务",
            "version": settings.app_version,
            "web_ui": "/upload",
            "docs": "/docs",
            "health": "/health"
        }


@app.get("/upload", response_class=HTMLResponse)
async def upload_page():
    """文件上传页面"""
    try:
        with open("templates/upload.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="上传页面未找到")


@app.get("/api", response_model=dict)
async def api_info():
    """API信息"""
    return {
        "message": "欢迎使用BERT语义分段服务API",
        "version": settings.app_version,
        "web_ui": "/upload",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "text_segment": "/segment",
            "markdown_upload": "/upload-markdown",
            "markdown_upload_async": "/upload-markdown-async",
            "task_status": "/task-status/{task_id}",
            "task_result": "/task-result/{task_id}",
            "config": "/config"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check(seg: SemanticSegmenter = Depends(get_segmenter)):
    """健康检查"""
    try:
        model_info = seg.bert_model.get_model_info()
        uptime = time.time() - startup_time if startup_time else 0
        
        return HealthResponse(
            status="healthy",
            version=settings.app_version,
            model_info=model_info,
            uptime=uptime
        )
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        raise HTTPException(status_code=503, detail="服务不健康")


@app.post("/segment", response_model=SegmentationResponse)
async def segment_text(
    request: SegmentationRequest,
    seg: SemanticSegmenter = Depends(get_segmenter)
):
    """
    文本语义分段
    
    对输入的文本进行语义分析，自动分割成语义连贯的段落。
    """
    start_time = time.time()
    
    try:
        # 如果请求中指定了阈值，临时设置
        original_threshold = seg.threshold
        if request.threshold is not None:
            seg.set_threshold(request.threshold)
        
        # 执行分段
        result = seg.segment_text(request.text)
        
        # 恢复原始阈值
        if request.threshold is not None:
            seg.set_threshold(original_threshold)
        
        # 检查分段结果
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        # 格式化输出
        separator = request.separator or settings.separator
        formatted_text = seg.text_processor.format_output(
            result["paragraphs"], separator
        )
        
        # 构建质量指标
        quality_data = result.get("quality", {})
        quality = QualityMetrics(
            paragraph_count=quality_data.get("paragraph_count", 0),
            avg_length=quality_data.get("avg_length", 0.0),
            length_std=quality_data.get("length_std", 0.0),
            too_short_count=quality_data.get("too_short_count", 0),
            too_long_count=quality_data.get("too_long_count", 0),
            semantic_consistency=quality_data.get("semantic_consistency", 0.0),
            quality_score=quality_data.get("quality_score", 0.0)
        )
        
        processing_time = time.time() - start_time
        
        return SegmentationResponse(
            success=True,
            paragraphs=result["paragraphs"],
            formatted_text=formatted_text,
            sentence_count=result.get("sentence_count", 0),
            boundary_count=result.get("boundary_count", 0),
            quality=quality,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"分段处理失败: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"分段处理失败: {str(e)}"
        )


@app.get("/config", response_model=ConfigResponse)
async def get_config(seg: SemanticSegmenter = Depends(get_segmenter)):
    """获取当前配置"""
    try:
        current_config = {
            "threshold": seg.threshold,
            "min_paragraph_length": seg.min_paragraph_length,
            "max_paragraph_length": seg.max_paragraph_length,
            "model_name": seg.bert_model.model_name,
            "device": seg.bert_model.device
        }
        
        return ConfigResponse(
            success=True,
            current_config=current_config,
            message="配置获取成功"
        )
    except Exception as e:
        logger.error(f"配置获取失败: {e}")
        raise HTTPException(status_code=500, detail="配置获取失败")


@app.put("/config", response_model=ConfigResponse)
async def update_config(
    request: ConfigRequest,
    seg: SemanticSegmenter = Depends(get_segmenter)
):
    """更新配置"""
    try:
        updated_fields = []
        
        # 更新阈值
        if request.threshold is not None:
            seg.set_threshold(request.threshold)
            updated_fields.append("threshold")
        
        # 更新段落长度限制
        if request.min_paragraph_length is not None:
            seg.min_paragraph_length = request.min_paragraph_length
            updated_fields.append("min_paragraph_length")
        
        if request.max_paragraph_length is not None:
            seg.max_paragraph_length = request.max_paragraph_length
            updated_fields.append("max_paragraph_length")
        
        # 获取更新后的配置
        current_config = {
            "threshold": seg.threshold,
            "min_paragraph_length": seg.min_paragraph_length,
            "max_paragraph_length": seg.max_paragraph_length
        }
        
        message = f"成功更新配置: {', '.join(updated_fields)}" if updated_fields else "无配置更新"
        
        return ConfigResponse(
            success=True,
            current_config=current_config,
            message=message
        )
        
    except Exception as e:
        logger.error(f"配置更新失败: {e}")
        raise HTTPException(status_code=500, detail=f"配置更新失败: {str(e)}")


def get_chunk_processor() -> ChunkProcessor:
    """获取分块处理器实例"""
    if chunk_processor is None:
        raise HTTPException(status_code=503, detail="分块处理器尚未初始化")
    return chunk_processor


@app.post("/upload-markdown", response_model=MarkdownProcessingResponse)
async def upload_markdown(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    preserve_structure: bool = Form(True),
    chunk_size: int = Form(5000),
    processor: ChunkProcessor = Depends(get_chunk_processor)
):
    """
    上传Markdown文件进行语义分段处理
    
    支持大文件智能分块处理，保持Markdown结构完整性。
    """
    start_time = time.time()
    
    try:
        # 验证文件类型
        if not file.filename:
            raise HTTPException(status_code=400, detail="文件名不能为空")
        
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ['.md', '.markdown']:
            return FileUploadErrorResponse(
                error="不支持的文件类型",
                error_type="ValidationError",
                filename=file.filename,
                details={
                    "supported_types": [".md", ".markdown"],
                    "received_type": file_ext
                }
            )
        
        # 验证文件大小 (50MB限制)
        max_size = 50 * 1024 * 1024  # 50MB
        content = await file.read()
        file_size = len(content)
        
        if file_size > max_size:
            return FileUploadErrorResponse(
                error="文件过大",
                error_type="FileSizeError",
                filename=file.filename,
                file_size=file_size,
                details={
                    "max_size": max_size,
                    "received_size": file_size
                }
            )
        
        # 验证参数
        if not (0.0 <= threshold <= 1.0):
            raise HTTPException(status_code=400, detail="阈值必须在0.0到1.0之间")
        
        if not (1000 <= chunk_size <= 20000):
            raise HTTPException(status_code=400, detail="分块大小必须在1000到20000之间")
        
        # 解码文件内容
        try:
            markdown_content = content.decode('utf-8')
        except UnicodeDecodeError:
            return FileUploadErrorResponse(
                error="文件编码错误，请确保文件为UTF-8编码",
                error_type="EncodingError",
                filename=file.filename,
                file_size=file_size
            )
        
        # 处理文档
        result = processor.process_markdown_sync(
            filename=file.filename,
            content=markdown_content,
            threshold=threshold,
            chunk_size=chunk_size,
            preserve_structure=preserve_structure
        )
        
        processing_time = time.time() - start_time
        
        if not result['success']:
            return FileUploadErrorResponse(
                error=result.get('error', '处理失败'),
                error_type="ProcessingError",
                filename=file.filename,
                file_size=file_size
            )
        
        # 构建响应
        return MarkdownProcessingResponse(
            success=True,
            filename=file.filename,
            file_size=file_size,
            processed_markdown=result['processed_markdown'],
            original_stats=MarkdownProcessingStats(**result['original_stats']),
            processed_stats=MarkdownProcessingStats(**result['processed_stats']),
            blocks_count=result['blocks_count'],
            processing_info=result['processing_info'],
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Markdown上传处理失败: {e}")
        return FileUploadErrorResponse(
            error=f"服务器内部错误: {str(e)}",
            error_type="InternalError",
            filename=file.filename if file.filename else "unknown"
        )


@app.post("/upload-markdown-async")
async def upload_markdown_async(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    preserve_structure: bool = Form(True),
    chunk_size: int = Form(5000),
    processor: ChunkProcessor = Depends(get_chunk_processor)
):
    """
    异步上传Markdown文件进行处理（适用于大文件）
    
    返回任务ID，可通过状态查询接口跟踪进度。
    """
    try:
        # 验证文件
        if not file.filename:
            raise HTTPException(status_code=400, detail="文件名不能为空")
        
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ['.md', '.markdown']:
            raise HTTPException(status_code=400, detail="不支持的文件类型")
        
        # 读取内容
        content = await file.read()
        file_size = len(content)
        
        # 大文件限制 (100MB)
        max_size = 100 * 1024 * 1024
        if file_size > max_size:
            raise HTTPException(status_code=400, detail="文件过大")
        
        # 解码内容
        try:
            markdown_content = content.decode('utf-8')
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="文件编码错误")
        
        # 创建处理任务
        task_id = processor.create_task(
            filename=file.filename,
            content=markdown_content,
            threshold=threshold,
            chunk_size=chunk_size,
            preserve_structure=preserve_structure
        )
        
        # 启动异步处理
        import asyncio
        asyncio.create_task(processor.process_markdown_async(task_id))
        
        return {
            "success": True,
            "task_id": task_id,
            "message": "任务已创建，正在处理中",
            "status_url": f"/task-status/{task_id}",
            "result_url": f"/task-result/{task_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"异步任务创建失败: {e}")
        raise HTTPException(status_code=500, detail=f"任务创建失败: {str(e)}")


@app.get("/task-status/{task_id}", response_model=ProcessingStatusResponse)
async def get_task_status(
    task_id: str,
    processor: ChunkProcessor = Depends(get_chunk_processor)
):
    """获取任务处理状态"""
    status = processor.get_task_status(task_id)
    
    if not status:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return ProcessingStatusResponse(**status)


@app.get("/task-result/{task_id}")
async def get_task_result(
    task_id: str,
    processor: ChunkProcessor = Depends(get_chunk_processor)
):
    """获取任务处理结果"""
    result = processor.get_task_result(task_id)
    
    if not result:
        status = processor.get_task_status(task_id)
        if not status:
            raise HTTPException(status_code=404, detail="任务不存在")
        elif status['status'] == "processing":
            raise HTTPException(status_code=202, detail="任务正在处理中")
        elif status['status'] == "failed":
            raise HTTPException(status_code=500, detail="任务处理失败")
        else:
            raise HTTPException(status_code=404, detail="结果不可用")
    
    return result


@app.get("/tasks")
async def list_tasks(processor: ChunkProcessor = Depends(get_chunk_processor)):
    """列出所有任务状态"""
    return {
        "tasks": processor.get_all_tasks_status()
    }


@app.delete("/task/{task_id}")
async def cancel_task(
    task_id: str,
    processor: ChunkProcessor = Depends(get_chunk_processor)
):
    """取消任务"""
    success = processor.cancel_task(task_id)
    
    if not success:
        raise HTTPException(status_code=400, detail="无法取消任务")
    
    return {"success": True, "message": "任务已取消"}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )