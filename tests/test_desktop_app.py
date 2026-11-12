import json
import tempfile
import unittest
from pathlib import Path

from eightfm import autostart
from eightfm.desktop_app import AppSettings, normalize_phone


class PhoneTests(unittest.TestCase):
    def test_phone_is_normalized_for_telethon(self):
        self.assertEqual(normalize_phone("+375 44 7224002"), "+375447224002")
        self.assertEqual(normalize_phone("375 (44) 722-40-02"), "+375447224002")
        self.assertEqual(normalize_phone("0044 1234 5678"), "+4412345678")
        self.assertEqual(normalize_phone(""), "")

    def test_assistant_config_gets_normalized_phone(self):
        settings = AppSettings(api_id=1, api_hash="h", phone="+375 44 7224002")
        config = settings.to_assistant_config(Path("C:/userdata"))
        self.assertEqual(config.phone, "+375447224002")


class AppSettingsTests(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            settings = AppSettings(
                api_id=123456,
                api_hash="secret",
                phone="+375447224002",
                player_name="Другой игрок",
                chat="@phys8math8bot",
                reply_delay_seconds=7.5,
                dry_run=True,
                autostart=False,
                start_minimized=False,
            )
            settings.save(path)
            self.assertEqual(AppSettings.load(path), settings)

    def test_model_defaults_to_lu_luna(self):
        self.assertEqual(AppSettings().model_path, "models/lu_luna.json")
        self.assertEqual(AppSettings(model_path="").to_assistant_config(Path("C:/userdata")).model_path, str(Path(__file__).resolve().parent.parent / "models" / "lu_luna.json"))

    def test_unknown_keys_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps({"player_name": "Аня", "unknown": 1}),
                encoding="utf-8",
            )
            self.assertEqual(AppSettings.load(path).player_name, "Аня")

    def test_star_reminder_flag_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            settings = AppSettings(star_reminder_shown=True)
            settings.save(path)
            self.assertTrue(AppSettings.load(path).star_reminder_shown)

    def test_config_with_bom_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                '\ufeff{"player_name": "Аня"}',
                encoding="utf-8",
            )
            self.assertEqual(AppSettings.load(path).player_name, "Аня")

    def test_seed_from_legacy_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            legacy = Path(tmp) / "telegram-config.json"
            legacy.write_text(
                json.dumps(
                    {
                        "api_id": 111,
                        "api_hash": "legacy-hash",
                        "phone": "+100",
                        "player_name": "Старый ник",
                        "model_path": "models/lu_luna.json",
                        "min_delay_minutes": 40,
                        "max_delay_minutes": 80,
                    }
                ),
                encoding="utf-8",
            )
            fresh = Path(tmp) / "app" / "config.json"
            settings = AppSettings.load_or_seed(fresh, [legacy])
            self.assertEqual(settings.api_id, 111)
            self.assertEqual(settings.player_name, "Старый ник")
            self.assertEqual(settings.reply_delay_seconds, 20.0)
            self.assertTrue(fresh.exists())

    def test_is_configured_requires_credentials(self):
        self.assertFalse(AppSettings().is_configured())
        self.assertFalse(AppSettings(api_id=1, api_hash="h").is_configured())
        self.assertTrue(
            AppSettings(api_id=1, api_hash="h", phone="+1").is_configured()
        )

    def test_to_assistant_config_uses_user_session_and_bundled_model(self):
        settings = AppSettings(api_id=1, api_hash="h", phone="+1", player_name="Аня")
        config = settings.to_assistant_config(Path("C:/userdata"))
        self.assertEqual(Path(config.session).name, "telegram-assistant")
        self.assertTrue(Path(config.model_path).name == "lu_luna.json")
        self.assertEqual(config.player_name, "Аня")
        self.assertEqual(config.reply_delay_seconds, 20.0)
        self.assertIsNone(config.seed)

    def test_absolute_model_path_is_preserved(self):
        settings = AppSettings(model_path="C:/models/custom.json")
        config = settings.to_assistant_config(Path("C:/userdata"))
        self.assertEqual(Path(config.model_path), Path("C:/models/custom.json"))


class AutostartCommandTests(unittest.TestCase):
    def test_launch_command_uses_launcher_and_flag(self):
        command = autostart.launch_command(minimized=True)
        self.assertIn("run_assistant_app.py", command)
        self.assertTrue(command.endswith("--minimized"))

    def test_launch_command_without_flag(self):
        command = autostart.launch_command(minimized=False)
        self.assertIn("run_assistant_app.py", command)
        self.assertFalse(command.endswith("--minimized"))


if __name__ == "__main__":
    unittest.main()
