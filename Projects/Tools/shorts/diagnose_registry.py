from pprint import pprint
from core.registry_loader import resolve_channel

resolved = resolve_channel("history")

print("\n=== RESOLVED CHANNEL RESULT ===")
print("Result type:", type(resolved).__name__)

if isinstance(resolved, dict):
    print("Top-level keys:", list(resolved.keys()))

    for key in ("channel", "worlds", "characters"):
        value = resolved.get(key)

        print("\n=== " + key.upper() + " ===")
        print("Container type:", type(value).__name__)

        if isinstance(value, (list, tuple)):
            print("Item count:", len(value))

            for index, item in enumerate(value):
                print("\nItem", index, "- type:", type(item).__name__)

                if isinstance(item, dict):
                    pprint(item)
                elif hasattr(item, "__dict__"):
                    pprint(vars(item))
                else:
                    pprint(item)
        elif isinstance(value, dict):
            pprint(value)
        elif hasattr(value, "__dict__"):
            pprint(vars(value))
        else:
            pprint(value)
else:
    pprint(resolved)
