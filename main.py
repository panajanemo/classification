#!/usr/bin/env python3
"""
BERT语义分段服务启动脚本
"""

import sys
import os
import logging

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """主函数"""
    try:
        import uvicorn
        from config.settings import settings
        
        logger.info(f"启动{settings.app_name} v{settings.app_version}")
        logger.info(f"服务地址: http://{settings.host}:{settings.port}")
        logger.info(f"API文档: http://{settings.host}:{settings.port}/docs")
        
        uvicorn.run(
            "src.api.app:app",
            host=settings.host,
            port=settings.port,
            reload=settings.debug,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        logger.info("服务已停止")
    except Exception as e:
        logger.error(f"服务启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()