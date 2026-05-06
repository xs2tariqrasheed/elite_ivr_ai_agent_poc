# Migration Guide - LLM Service Refactoring

## What Changed
The monolithic `services/llm_service.py` has been split into modular files under `services/llm/` directory.

## For Existing Code
**No changes needed!** Existing imports continue to work:

```python
# This still works exactly as before:
from services import llm_service as llm

llm.extract_account_number(text)
llm.extract_phone_number(text)
llm.warm_up_model()
```

## For New Code
You can now import specific functions:

```python
# Option 1: Import the module (backward compatible)
from services import llm

# Option 2: Import specific functions
from services.llm import extract_account_number, extract_phone_number, warm_up_model

# Option 3: Import from submodules (for direct access)
from services.llm.account_number import extract_account_number
from services.llm.phone_number import extract_phone_number
from services.llm.warm_up import warm_up_model
from services.llm.llm_client import get_ollama_client
```

## File Mapping (Old → New)

| Functionality | Old Location | New Location |
|---------------|--------------|--------------|
| Ollama client | `llm_service.py` | `services/llm/llm_client.py` |
| Account extraction | `llm_service.py` | `services/llm/account_number.py` |
| Phone extraction | `llm_service.py` | `services/llm/phone_number.py` |
| Model warm-up | `llm_service.py` | `services/llm/warm_up.py` |
| Helper functions | `llm_service.py` | `services/llm/_helpers.py` |

## Testing Imports
To verify the refactoring works in your environment:

```python
# Test imports
from services import llm

# Verify all functions exist
assert hasattr(llm, 'extract_account_number')
assert hasattr(llm, 'extract_phone_number')
assert hasattr(llm, 'warm_up_model')
assert hasattr(llm, 'get_ollama_client')

print("✓ All imports working correctly!")
```

## When to Delete Old File
You can safely delete `services/llm_service.py` after:
1. Running all tests and confirming they pass
2. Verifying the application runs without errors
3. Checking that all imports resolve correctly

## Questions?
The refactoring maintains 100% backward compatibility. If you encounter any import issues, verify that:
- The `services/llm/` directory exists
- All Python files in `services/llm/` are present and readable
- Your Python path includes the workspace directory
