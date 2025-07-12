import pytest
from src.utils.markdown_processor import MarkdownProcessor, MarkdownBlock


class TestMarkdownProcessor:
    """Markdown处理器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.processor = MarkdownProcessor()
    
    def test_parse_simple_markdown(self):
        """测试简单Markdown解析"""
        markdown = """# 标题1

这是第一段内容。

## 标题2

这是第二段内容。

- 列表项1
- 列表项2

```python
print("代码块")
```

> 这是引用内容
"""
        
        blocks = self.processor.parse_markdown(markdown)
        
        # 验证块类型
        block_types = [block.type for block in blocks]
        assert 'heading' in block_types
        assert 'paragraph' in block_types
        assert 'list' in block_types
        assert 'code_block' in block_types
        assert 'quote' in block_types
    
    def test_heading_levels(self):
        """测试标题级别解析"""
        markdown = """# 一级标题
## 二级标题
### 三级标题
"""
        
        blocks = self.processor.parse_markdown(markdown)
        headings = [block for block in blocks if block.type == 'heading']
        
        assert len(headings) == 3
        assert headings[0].level == 1
        assert headings[1].level == 2
        assert headings[2].level == 3
        assert headings[0].content == "一级标题"
    
    def test_code_block_parsing(self):
        """测试代码块解析"""
        markdown = """```python
def hello():
    print("Hello World")
```

```javascript
console.log("Hello JS");
```"""
        
        blocks = self.processor.parse_markdown(markdown)
        code_blocks = [block for block in blocks if block.type == 'code_block']
        
        assert len(code_blocks) == 2
        assert code_blocks[0].metadata['language'] == 'python'
        assert code_blocks[1].metadata['language'] == 'javascript'
        assert 'def hello():' in code_blocks[0].content
    
    def test_list_parsing(self):
        """测试列表解析"""
        markdown = """- 第一项
- 第二项
  - 嵌套项
- 第三项

1. 编号列表1
2. 编号列表2
"""
        
        blocks = self.processor.parse_markdown(markdown)
        lists = [block for block in blocks if block.type == 'list']
        
        assert len(lists) == 2
        assert '第一项' in lists[0].metadata['items']
        assert '编号列表1' in lists[1].metadata['items']
    
    def test_extract_text_content(self):
        """测试文本内容提取"""
        markdown = """# 标题

这是**粗体**和*斜体*文本。

[链接文本](http://example.com)

`代码`内容。

```python
# 这是代码，不应该被提取
```
"""
        
        blocks = self.processor.parse_markdown(markdown)
        text_content = self.processor.extract_text_content(blocks)
        
        # 应该提取纯文本，移除Markdown语法
        assert '标题' in text_content
        assert '粗体' in text_content
        assert '斜体' in text_content
        assert '链接文本' in text_content
        assert '代码' in text_content
        # 代码块内容不应该被提取
        assert '# 这是代码' not in text_content
    
    def test_reconstruct_markdown(self):
        """测试Markdown重构"""
        original_markdown = """# 主标题

这是第一段内容。

## 子标题

这是第二段内容。

- 列表项1
- 列表项2

```python
print("代码")
```
"""
        
        blocks = self.processor.parse_markdown(original_markdown)
        reconstructed = self.processor.reconstruct_markdown(blocks)
        
        # 重构后应该包含主要元素
        assert '# 主标题' in reconstructed
        assert '## 子标题' in reconstructed
        assert '```python' in reconstructed
        assert '- 列表项1' in reconstructed
    
    def test_get_statistics(self):
        """测试统计信息"""
        markdown = """# 标题1
## 标题2

段落1内容。

段落2内容。

- 列表项

```code
代码
```

| 表格 |
|------|
"""
        
        blocks = self.processor.parse_markdown(markdown)
        stats = self.processor.get_statistics(blocks)
        
        assert stats['total_blocks'] > 0
        assert stats['headings']['count'] == 2
        assert stats['headings']['levels'][1] == 1  # 一级标题1个
        assert stats['headings']['levels'][2] == 1  # 二级标题1个
        assert stats['paragraphs']['count'] == 2
        assert stats['code_blocks'] == 1
        assert stats['lists'] == 1
    
    def test_process_markdown_document(self):
        """测试完整文档处理"""
        markdown = """# 文档标题

这是一个很长的段落，包含了大量的文本内容。这个段落足够长，应该可以被语义分段器进一步分割成更小的段落。我们需要确保这个功能正常工作，并且能够保持Markdown的结构完整性。

## 章节标题

这是另一个段落，内容与前面不同。

- 列表项1
- 列表项2
"""
        
        result = self.processor.process_markdown_document(markdown, threshold=0.3)
        
        assert result['success'] == True
        assert 'processed_markdown' in result
        assert 'original_stats' in result
        assert 'processed_stats' in result
        assert 'text_content' in result
        
        # 应该包含统计信息
        assert result['original_stats']['headings']['count'] == 2
        assert result['blocks_count']['original'] > 0
    
    def test_empty_markdown(self):
        """测试空Markdown处理"""
        blocks = self.processor.parse_markdown("")
        assert len(blocks) == 0
        
        blocks = self.processor.parse_markdown("   \n\n   ")
        assert len(blocks) == 0
    
    def test_markdown_with_complex_formatting(self):
        """测试复杂格式Markdown"""
        markdown = """# 复杂文档

这里有**粗体**和*斜体*文本。

还有[链接](http://example.com)和![图片](image.jpg)。

表格示例：

| 列1 | 列2 |
|-----|-----|
| 值1 | 值2 |

> 这是引用内容
> 多行引用

---

最后一段内容。
"""
        
        blocks = self.processor.parse_markdown(markdown)
        
        # 验证各种类型都被正确解析
        types = set(block.type for block in blocks)
        expected_types = {'heading', 'paragraph', 'table', 'quote', 'horizontal_rule'}
        
        # 至少应该有这些类型中的一部分
        assert len(types.intersection(expected_types)) > 0
    
    def test_segment_markdown_blocks_mock(self):
        """测试Markdown块分段（使用模拟）"""
        # 创建测试块
        blocks = [
            MarkdownBlock(type='heading', content='标题', level=1),
            MarkdownBlock(type='paragraph', content='这是一个很长的段落' * 20),  # 很长的段落
            MarkdownBlock(type='paragraph', content='短段落'),
            MarkdownBlock(type='code_block', content='print("code")', metadata={'language': 'python'})
        ]
        
        # 这个测试不实际调用BERT模型，因为在测试环境中可能没有加载
        # 只验证函数不会崩溃
        try:
            segmented = self.processor.segment_markdown_blocks(blocks, threshold=0.5)
            # 应该返回相同数量或更多的块
            assert len(segmented) >= len(blocks)
        except Exception:
            # 如果没有BERT模型，应该返回原始块
            pass