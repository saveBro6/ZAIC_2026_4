"""Task-Aligned Schema Linking module based on 'Automatic Metadata Extraction for Text-to-SQL'."""

from zaic_2026_4.schema_linking.ast_parser import SQLASTExtractor
from zaic_2026_4.schema_linking.context_builder import (
    SchemaContextBuilder,
    SchemaVariant,
)
from zaic_2026_4.schema_linking.post_processor import (
    SchemaPostProcessor,
    normalize_identifier,
)
from zaic_2026_4.schema_linking.task_aligned_linker import (
    SchemaLinkingResult,
    TaskAlignedSchemaLinker,
    VariantExecutionTrace,
    task_aligned_schema_linking,
)

__all__ = [
    "SQLASTExtractor",
    "SchemaContextBuilder",
    "SchemaLinkingResult",
    "SchemaPostProcessor",
    "SchemaVariant",
    "TaskAlignedSchemaLinker",
    "VariantExecutionTrace",
    "normalize_identifier",
    "task_aligned_schema_linking",
]
