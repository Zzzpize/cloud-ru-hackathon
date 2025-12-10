import pytest
import sys

from pathlib import Path
from unittest.mock import AsyncMock, patch
from tools import create_or_update_checklist, get_checklist, list_checklists, delete_checklist
from mcp.server.fastmcp import Context

sys.path.insert(0, str(Path(__file__).parent / "src"))

@pytest.mark.asyncio
async def test_create_checklist():
    """Тест создания чек-листа"""
    ctx = AsyncMock(spec=Context)
    
    criteria_data = [{
        "criterion_id": "c1",
        "category": "quality",
        "text": "Проверить качество",
        "is_mandatory": True,
        "check_type": "keyword",
        "keywords": ["качество", "стандарт"]
    }]
    
    result = await create_or_update_checklist(
        ctx=ctx,
        tenant_id="tenant_1",
        checklist_id="checklist_1",
        name="Проверка качества",
        criteria=criteria_data
    )
    
    assert result.content is not None
    assert len(result.content) > 0


@pytest.mark.asyncio
async def test_get_checklist():
    """Тест получения чек-листа"""
    ctx = AsyncMock(spec=Context)
    
    # Сначала создаём чек-лист
    criteria_data = [{
        "criterion_id": "c1",
        "category": "quality",
        "text": "Проверить качество",
        "is_mandatory": True,
        "check_type": "keyword",
        "keywords": []
    }]
    
    await create_or_update_checklist(
        ctx=ctx,
        tenant_id="test_tenant",
        checklist_id="test_checklist",
        name="Тестовый чек-лист",
        criteria=criteria_data
    )
    
    # Затем получаем его
    result = await get_checklist(
        ctx=ctx,
        tenant_id="test_tenant",
        checklist_id="test_checklist"
    )
    
    assert result.content is not None


@pytest.mark.asyncio
async def test_list_checklists():
    """Тест получения списка чек-листов"""
    ctx = AsyncMock(spec=Context)
    
    result = await list_checklists(ctx=ctx, tenant_id="test_tenant")
    
    assert result.content is not None
    assert len(result.content) > 0


@pytest.mark.asyncio
async def test_delete_checklist():
    """Тест удаления чек-листа"""
    ctx = AsyncMock(spec=Context)
    
    criteria_data = [{
        "criterion_id": "c1",
        "category": "quality",
        "text": "Проверить",
        "is_mandatory": True,
        "check_type": "keyword",
        "keywords": []
    }]
    
    # Создаём чек-лист
    await create_or_update_checklist(
        ctx=ctx,
        tenant_id="del_tenant",
        checklist_id="del_checklist",
        name="Для удаления",
        criteria=criteria_data
    )
    
    # Удаляем его
    result = await delete_checklist(
        ctx=ctx,
        tenant_id="del_tenant",
        checklist_id="del_checklist"
    )
    
    assert result.content is not None