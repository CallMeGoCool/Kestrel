"""
Kestrel Shorts - Registry Loader
================================

The registry answers "what exists?". Each JSON file describes one thing and
references other things by ID:

    Channel   -> world_ids, available_character_ids
    Character -> voice_profile_id, avatar_id
    Avatar    -> canonical image + explicit pose list
    Voice     -> provider + reference clip (relative to F5_TTS_MODEL_DIR)

Views
-----
Menu views (id + name + description) are what agents see while choosing.
Full views are loaded only for the thing that was chosen.

    channel_menu()            -> every channel                (Orchestrator)
    world_menu(channel_id)    -> that channel's worlds        (Script Refiner)
    character_menu(channel_id)-> that channel's characters    (Story Director)
    pose_menu(character_id)   -> valid pose IDs               (Visual Director)

Run `python -m core.registry_loader` to validate every file.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from core.models._serde import SchemaError, from_dict
from core.models.avatar import Avatar
from core.models.channel import Channel
from core.models.character import Character
from core.models.theme import Theme
from core.models.voice_profile import VoiceProfile
from core.models.world import World

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_ROOT = PROJECT_ROOT / "core" / "registry"

# category folder -> (id field, model)
CATEGORIES = {
    "channels": ("channel_id", Channel),
    "worlds": ("world_id", World),
    "characters": ("character_id", Character),
    "avatars": ("avatar_id", Avatar),
    "voices": ("voice_profile_id", VoiceProfile),
    "themes": ("theme_id", Theme),
}


class RegistryError(Exception):
    """Raised when registry configuration is missing or invalid."""


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


class RegistryLoader:
    def __init__(
        self,
        registry_root: str | Path | None = None,
        project_root: str | Path | None = None,
        f5_model_dir: str | Path | None = None,
    ) -> None:
        self.registry_root = (
            Path(registry_root).resolve() if registry_root else REGISTRY_ROOT
        )
        self.project_root = (
            Path(project_root).resolve() if project_root else PROJECT_ROOT
        )
        self._f5_model_dir = Path(f5_model_dir) if f5_model_dir else None
        self._cache: dict[str, tuple[dict[str, dict], list[str]]] = {}

    def reload(self) -> None:
        """Forget cached files; call after adding or editing registry JSON."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            with path.open("r", encoding="utf-8-sig") as file:
                data = json.load(file)
        except json.JSONDecodeError as exc:
            raise RegistryError(f"Invalid JSON in {path.name}: {exc}") from exc
        except OSError as exc:
            raise RegistryError(f"Could not read {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise RegistryError(f"{path.name} must contain a JSON object.")
        return data

    def _scan(self, category: str) -> tuple[dict[str, dict], list[str]]:
        """Read one category, collecting problems instead of stopping."""
        if category not in self._cache:
            id_field = CATEGORIES[category][0]
            directory = self.registry_root / category
            entries: dict[str, dict] = {}
            problems: list[str] = []

            if directory.is_dir():
                for path in sorted(directory.glob("*.json")):
                    try:
                        raw = self._read_json(path)
                    except RegistryError as exc:
                        problems.append(f"{category}/{exc}")
                        continue

                    entry_id = raw.get(id_field)
                    if not isinstance(entry_id, str) or not entry_id.strip():
                        problems.append(
                            f"{category}/{path.name}: missing or invalid {id_field!r}"
                        )
                        continue
                    if entry_id in entries:
                        problems.append(
                            f"{category}/{path.name}: duplicate {id_field} {entry_id!r}"
                        )
                        continue
                    entries[entry_id] = raw

            self._cache[category] = (entries, problems)
        return self._cache[category]

    def _entries(self, category: str) -> dict[str, dict]:
        if category not in CATEGORIES:
            raise RegistryError(f"Unknown registry category: {category!r}")
        entries, problems = self._scan(category)
        if problems:
            raise RegistryError("; ".join(problems))
        return entries

    def list_entries(self, category: str) -> list[dict[str, Any]]:
        return [copy.deepcopy(e) for e in self._entries(category).values()]

    def get_entry(self, category: str, entry_id: str) -> dict[str, Any]:
        if not isinstance(entry_id, str) or not entry_id.strip():
            raise RegistryError("Registry entry ID must be a non-empty string.")
        entries = self._entries(category)
        if entry_id not in entries:
            raise RegistryError(f"No {category} entry found with ID {entry_id!r}")
        return copy.deepcopy(entries[entry_id])

    # Raw (dict) getters, kept for existing callers.
    def get_channel(self, channel_id: str) -> dict[str, Any]:
        return self.get_entry("channels", channel_id)

    def get_world(self, world_id: str) -> dict[str, Any]:
        return self.get_entry("worlds", world_id)

    def get_character(self, character_id: str) -> dict[str, Any]:
        return self.get_entry("characters", character_id)

    def get_avatar(self, avatar_id: str) -> dict[str, Any]:
        return self.get_entry("avatars", avatar_id)

    def get_voice(self, voice_profile_id: str) -> dict[str, Any]:
        return self.get_entry("voices", voice_profile_id)

    def get_theme(self, theme_id: str) -> dict[str, Any]:
        return self.get_entry("themes", theme_id)

    def load_theme(self, theme_id: str) -> Theme:
        return from_dict(Theme, self.get_theme(theme_id))

    # ------------------------------------------------------------------
    # Typed getters
    # ------------------------------------------------------------------

    def load_channel(self, channel_id: str) -> Channel:
        return from_dict(Channel, self.get_channel(channel_id))

    def load_world(self, world_id: str) -> World:
        return from_dict(World, self.get_world(world_id))

    def load_avatar(self, avatar_id: str) -> Avatar:
        return from_dict(Avatar, self.get_avatar(avatar_id))

    def load_voice(self, voice_profile_id: str) -> VoiceProfile:
        return from_dict(VoiceProfile, self.get_voice(voice_profile_id))

    def load_character(self, character_id: str, resolve: bool = True) -> Character:
        """Typed character; with resolve=True its voice and avatar are attached."""
        raw = self.get_character(character_id)
        if resolve:
            if raw.get("voice_profile_id"):
                raw["voice"] = self.get_voice(raw["voice_profile_id"])
            if raw.get("avatar_id"):
                raw["avatar"] = self.get_avatar(raw["avatar_id"])
        return from_dict(Character, raw)

    # ------------------------------------------------------------------
    # Menu views
    # ------------------------------------------------------------------

    def channel_menu(self) -> list[dict[str, str]]:
        return [
            {"id": e["channel_id"], "name": e.get("name", ""),
             "description": e.get("description", "")}
            for e in self._entries("channels").values()
        ]

    def list_channels(self) -> list[dict[str, Any]]:
        """Lightweight channel records (kept for existing callers)."""
        keep = ("channel_id", "name", "description", "family_id", "world_ids",
                "storytelling_style", "available_character_ids")
        return [
            {k: e[k] for k in keep if k in e}
            for e in self._entries("channels").values()
        ]

    def world_menu(self, channel_id: str) -> list[dict[str, str]]:
        channel = self.get_channel(channel_id)
        result = []
        for world_id in _as_list(channel.get("world_ids")):
            world = self.get_world(world_id)
            result.append({"id": world_id, "name": world.get("name", ""),
                           "description": world.get("description", "")})
        return result

    def character_menu(self, channel_id: str) -> list[dict[str, str]]:
        channel = self.get_channel(channel_id)
        result = []
        for character_id in _as_list(channel.get("available_character_ids")):
            character = self.get_character(character_id)
            result.append({"id": character_id,
                           "name": character.get("display_name", ""),
                           "description": character.get("description", "")})
        return result

    def pose_menu(self, character_id: str) -> list[dict[str, Any]]:
        """Valid pose IDs for a character; the only poses a director may use."""
        character = self.get_character(character_id)
        avatar_id = character.get("avatar_id")
        if not avatar_id:
            return []
        avatar = self.get_avatar(avatar_id)
        default = avatar.get("default_pose")
        return [
            {"pose_id": p.get("pose_id"), "description": p.get("description", ""),
             "default": p.get("pose_id") == default}
            for p in _as_list(avatar.get("poses")) if isinstance(p, dict)
        ]

    # ------------------------------------------------------------------
    # Channel resolution (dict form, as before)
    # ------------------------------------------------------------------

    def resolve_channel(self, channel_id: str) -> dict[str, Any]:
        """
        Channel -> worlds and characters; character -> voice and avatar.
        Each resolved character carries `voice` and `avatar` dicts.
        """
        channel = self.get_channel(channel_id)
        worlds = [self.get_world(w) for w in _as_list(channel.get("world_ids"))]

        characters: list[dict[str, Any]] = []
        avatars: list[dict[str, Any]] = []
        seen_avatars: set[str] = set()

        for character_id in _as_list(channel.get("available_character_ids")):
            character = self.get_character(character_id)
            resolved = dict(character)
            resolved["voice"] = None
            resolved["avatar"] = None

            voice_id = character.get("voice_profile_id")
            if voice_id:
                voice = self.get_voice(voice_id)
                if voice.get("character_id") not in (None, character_id):
                    raise RegistryError(
                        f"Character {character_id!r} uses voice {voice_id!r}, "
                        f"which belongs to {voice.get('character_id')!r}."
                    )
                resolved["voice"] = voice

            avatar_id = character.get("avatar_id")
            if avatar_id:
                avatar = self.get_avatar(avatar_id)
                if avatar.get("character_id") not in (None, character_id):
                    raise RegistryError(
                        f"Character {character_id!r} uses avatar {avatar_id!r}, "
                        f"which belongs to {avatar.get('character_id')!r}."
                    )
                resolved["avatar"] = avatar
                if avatar_id not in seen_avatars:
                    seen_avatars.add(avatar_id)
                    avatars.append(avatar)

            characters.append(resolved)

        return {"channel": channel, "worlds": worlds,
                "characters": characters, "avatars": avatars}

    validate_channel = resolve_channel

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _f5_dir(self) -> Path | None:
        if self._f5_model_dir is not None:
            return self._f5_model_dir
        try:
            from core import config
            return Path(config.F5_TTS_MODEL_DIR)
        except Exception:
            return None

    def validate_all(self) -> dict[str, list[str]]:
        """
        Check every registry file and every cross-reference.
        Returns {"errors": [...], "warnings": [...]}; never raises.
        """
        errors: list[str] = []
        warnings: list[str] = []
        scanned: dict[str, dict[str, dict]] = {}

        for category, (_, model) in CATEGORIES.items():
            entries, problems = self._scan(category)
            errors.extend(problems)
            scanned[category] = entries
            for entry_id, raw in entries.items():
                local_warnings: list[str] = []
                try:
                    from_dict(model, raw, local_warnings)
                except SchemaError as exc:
                    errors.extend(f"{category}/{entry_id}: {p}" for p in exc.problems)
                warnings.extend(f"{category}/{entry_id}: {w}" for w in local_warnings)

        channels, worlds = scanned["channels"], scanned["worlds"]
        characters, avatars = scanned["characters"], scanned["avatars"]
        voices = scanned["voices"]

        for cid, ch in channels.items():
            for key, pool in (("world_ids", worlds),
                              ("available_character_ids", characters)):
                ids = _as_list(ch.get(key))
                if len(ids) != len(set(ids)):
                    errors.append(f"channels/{cid}: duplicate IDs in {key}")
                for ref in ids:
                    if ref not in pool:
                        errors.append(f"channels/{cid}: {key} references unknown {ref!r}")

        themes = scanned["themes"]
        for cid, ch in channels.items():
            tid = ch.get("theme_id")
            if not tid:
                warnings.append(f"channels/{cid}: no theme_id")
            elif tid not in themes:
                errors.append(f"channels/{cid}: theme_id references unknown {tid!r}")
        try:
            from creative.catalogs import visual_catalog as vc
            catalog = {"transitions": set(vc.TRANSITIONS), "overlays": set(vc.OVERLAYS)}
            for tid, t in themes.items():
                for key, known in catalog.items():
                    for name in _as_list(t.get(key)):
                        if name not in known:
                            warnings.append(f"themes/{tid}: unknown {key[:-1]} {name!r}")
        except Exception:
            pass

        for cid, c in characters.items():
            for key, pool, label in (("voice_profile_id", voices, "voice"),
                                     ("avatar_id", avatars, "avatar")):
                ref = c.get(key)
                if not ref:
                    warnings.append(f"characters/{cid}: no {key}")
                elif ref not in pool:
                    errors.append(f"characters/{cid}: {key} references unknown {ref!r}")
                elif pool[ref].get("character_id") != cid:
                    errors.append(
                        f"characters/{cid}: {label} {ref!r} belongs to "
                        f"{pool[ref].get('character_id')!r}"
                    )
            for rel in _as_list(c.get("relationships")):
                target = rel.get("character_id") if isinstance(rel, dict) else None
                if target not in characters:
                    errors.append(f"characters/{cid}: relationship to unknown {target!r}")

        for aid, a in avatars.items():
            poses = [p for p in _as_list(a.get("poses")) if isinstance(p, dict)]
            ids = [p.get("pose_id") for p in poses]
            if len(ids) != len(set(ids)):
                errors.append(f"avatars/{aid}: duplicate pose_id")
            if a.get("default_pose") not in ids:
                errors.append(f"avatars/{aid}: default_pose {a.get('default_pose')!r} is not in poses")
            assets = a.get("assets") if isinstance(a.get("assets"), dict) else {}
            canonical = assets.get("canonical_image")
            if canonical and not (self.project_root / canonical).is_file():
                errors.append(f"avatars/{aid}: canonical image not found: {canonical}")
            pose_dir = assets.get("poses_directory")
            if pose_dir:
                for p in poses:
                    if not (self.project_root / pose_dir / str(p.get("file"))).is_file():
                        errors.append(f"avatars/{aid}: pose file not found: {pose_dir}/{p.get('file')}")

        f5_dir = self._f5_dir()
        for vid, v in voices.items():
            ref = v.get("reference_audio")
            if ref:
                if f5_dir is None:
                    warnings.append(f"voices/{vid}: F5-TTS folder unknown; clip not checked")
                elif not (Path(ref) if Path(ref).is_absolute() else f5_dir / ref).is_file():
                    errors.append(f"voices/{vid}: reference clip not found: {ref}")
            if not v.get("ref_text"):
                warnings.append(f"voices/{vid}: empty ref_text (F5-TTS will auto-transcribe)")

        return {"errors": errors, "warnings": warnings}


_default_loader = RegistryLoader()


def reload() -> None:
    _default_loader.reload()

def list_channels(): return _default_loader.list_channels()
def get_channel(channel_id): return _default_loader.get_channel(channel_id)
def get_world(world_id): return _default_loader.get_world(world_id)
def get_character(character_id): return _default_loader.get_character(character_id)
def get_avatar(avatar_id): return _default_loader.get_avatar(avatar_id)
def get_voice(voice_profile_id): return _default_loader.get_voice(voice_profile_id)
def resolve_channel(channel_id): return _default_loader.resolve_channel(channel_id)
def channel_menu(): return _default_loader.channel_menu()
def world_menu(channel_id): return _default_loader.world_menu(channel_id)
def character_menu(channel_id): return _default_loader.character_menu(channel_id)
def pose_menu(character_id): return _default_loader.pose_menu(character_id)
def validate_all(): return _default_loader.validate_all()


if __name__ == "__main__":
    report = validate_all()
    for line in report["errors"]:
        print("ERROR  ", line)
    for line in report["warnings"]:
        print("warning", line)
    print(f"\n{len(report['errors'])} error(s), {len(report['warnings'])} warning(s)")
    raise SystemExit(1 if report["errors"] else 0)