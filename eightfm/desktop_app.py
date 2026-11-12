"""Desktop GUI + system-tray wrapper for the 8ФМ Telegram assistant.

Run it with ``python run_assistant_app.py`` or ``python -m eightfm app``.
The window collects the Telegram credentials and the player name, the assistant
itself runs in a background thread, and the app keeps working from the system
tray.  Autostart is registered per user through :mod:`eightfm.autostart`.
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import shutil
import sys
import threading
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from . import autostart
from .telegram_assistant import (
    AssistantConfig,
    TelegramAssistant,
    set_log_sink,
)

APP_NAME = "8ФМ Ассистент"
APP_VERSION = "1.0"
APP_DIR_NAME = "8FM Assistant"
DEFAULT_MODEL = "models/lu_luna.json"

BG = "#eef1f7"
CARD = "#ffffff"
CARD_SOFT = "#f8fafc"
FG = "#0f172a"
MUTED = "#64748b"
BORDER = "#e2e8f0"
ACCENT = "#2563eb"
ACCENT_HOVER = "#1d4ed8"
ACCENT_SOFT = "#dbeafe"
DARK = "#0b1220"
LOG_FG = "#cbd5e1"
OK = "#16a34a"
WARN = "#d97706"
ERROR = "#dc2626"

REPO_URL = "https://github.com/leonka123/8FM_AI"

GUIDE_TEXT = """КАК ПОДКЛЮЧИТЬ АССИСТЕНТА — ПОШАГОВО

1. Получите api_id и api_hash
   • Откройте https://my.telegram.org и войдите по номеру телефона.
   • Перейдите в «API development tools».
   • Заполните название приложения (например, 8FM Assistant),
     платформу оставьте Desktop, остальные поля можно не трогать.
   • Нажмите «Create application».
   • Скопируйте App api_id (число) и App api_hash (длинная строка).
   • api_hash — это секрет. Не публикуйте его и не пересылайте.

2. Заполните поля в приложении
   • api_id — число из my.telegram.org.
   • api_hash — строка оттуда же.
   • Телефон — номер аккаунта Telegram в формате +375447224002.
     Пробелы и скобки можно не убирать, приложение нормализует номер.
   • Ваш ник в игре — ровно тот, который видит игровой бот.
     У других игроков он другой, поэтому поле редактируемое.

3. Укажите игру
   • Чат с ботом — @phys8math8bot.
   • Пауза перед ходом — 20 секунд по умолчанию. Меньше делать не стоит:
     мгновенный ответ выглядит подозрительно.
   • Модель — оставьте пустым, чтобы использовать встроенную
     models/lu_luna.json, или выберите другой JSON через «Обзор…».

4. Войдите в Telegram
   • Нажмите «Запустить».
   • Telegram пришлёт код в чат «Telegram» (не в SMS).
   • Введите код в диалоге приложения.
   • Если включена двухэтапная проверка (2FA), введите пароль.
   • Сессия сохранится: при следующих запусках код не нужен.

5. Включите автозапуск и трей
   • Галочка «Автозапуск при входе» уже включена.
   • «Старт в трее» — приложение запускается без окна.
   • Крестик закрытия окна сворачивает приложение в трей.
   • Полный выход — только через меню иконки в трее → «Выход».

6. Проверьте работу
   • Статус в шапке должен стать зелёным: «Работает — жду ходы».
   • В журнале появятся строки о найденных сообщениях и выборе lu_luna.
   • Для безопасной проверки включите «Пробный режим»: ассистент
     покажет решение, но не нажмёт кнопку.

ЕСЛИ ЧТО-ТО НЕ РАБОТАЕТ
   • «database is locked» — закрыт второй экземпляр ассистента или старой
     консольной версии. Завершите его и запустите приложение снова.
   • Код не приходит — проверьте, что номер указан с кодом страны,
     и запросите код ещё раз. Telegram может ограничить частые запросы.
   • «api_id должен быть числом» — скопируйте только цифры.
   • Ошибка входа — удалите папку session в %APPDATA%\\8FM Assistant
     и войдите заново (это сбросит только сессию, не настройки).
   • Не нажимает кнопку — проверьте, что это 8ФМ Classic без
     модификаторов, ник совпадает, а в сообщении есть кнопки выбора.
   • Антивирус/SmartScreen ругается на .exe — это типично для
     неподписанных сборок PyInstaller: «Подробнее» → «Выполнить всё равно».

ГДЕ ЛЕЖАТ ДАННЫЕ
   %APPDATA%\\8FM Assistant\\config.json          настройки
   %APPDATA%\\8FM Assistant\\session\\             сессия Telegram
   %APPDATA%\\8FM Assistant\\logs\\assistant.log   журнал

Файл session даёт доступ к аккаунту — не пересылайте его никому.
"""


class LoginCancelled(Exception):
    """Raised when the user closes the Telegram login dialog."""


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def user_data_dir() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config"))
    path = base / APP_DIR_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def resource_path(relative: str) -> Path:
    """Resolve a bundled resource for both source and PyInstaller builds."""
    if getattr(sys, "frozen", False):
        root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    else:
        root = Path(__file__).resolve().parent.parent
    return root / relative


def normalize_phone(value: str) -> str:
    """Turn ``+375 44 7224002`` into ``+375447224002`` for Telethon."""
    cleaned = "".join(ch for ch in (value or "") if ch.isdigit() or ch == "+")
    if cleaned.startswith("00"):
        cleaned = "+" + cleaned[2:]
    if cleaned and not cleaned.startswith("+"):
        cleaned = "+" + cleaned
    return cleaned


def legacy_config_paths() -> list[Path]:
    roots: list[Path] = []
    if getattr(sys, "frozen", False):
        roots.append(Path(sys.executable).resolve().parent)
    else:
        roots.append(Path(__file__).resolve().parent.parent)
    roots.append(Path.cwd())
    found: list[Path] = []
    for root in roots:
        candidate = root / "telegram-config.json"
        if candidate not in found:
            found.append(candidate)
    return found


def _migrate_legacy_session(
    legacy_config: Path,
    data: dict[str, Any],
    data_dir: Path,
) -> None:
    """Copy the old Telethon session so the user does not log in again."""
    raw = data.get("session")
    if not raw:
        return
    source = Path(str(raw))
    if not source.is_absolute():
        source = legacy_config.resolve().parent / source
    source = source.with_suffix(".session")
    target = data_dir / "session" / "telegram-assistant.session"
    if source.exists() and not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(source, target)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
@dataclass
class AppSettings:
    """:class:`AssistantConfig` plus UI/tray preferences."""

    api_id: int | None = None
    api_hash: str = ""
    phone: str = ""
    chat: str = "@phys8math8bot"
    player_name: str = "Леончик"
    model_path: str = DEFAULT_MODEL
    reply_delay_seconds: float = 20.0
    dry_run: bool = False
    autostart: bool = True
    start_minimized: bool = True
    star_reminder_shown: bool = False

    def is_configured(self) -> bool:
        return bool(self.api_id and self.api_hash and self.phone)

    def to_assistant_config(self, data_dir: Path) -> AssistantConfig:
        model = self.model_path.strip() or DEFAULT_MODEL
        model_path = Path(model)
        if not model_path.is_absolute():
            model_path = resource_path(model)
        return AssistantConfig(
            api_id=self.api_id,
            api_hash=self.api_hash or None,
            phone=normalize_phone(self.phone) or None,
            chat=self.chat or "@phys8math8bot",
            player_name=self.player_name or "Леончик",
            model_path=str(model_path),
            session=str(data_dir / "session" / "telegram-assistant"),
            reply_delay_seconds=self.reply_delay_seconds,
            dry_run=self.dry_run,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "AppSettings":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        known = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in data.items() if key in known})

    @classmethod
    def load_or_seed(
        cls,
        path: Path,
        legacy_paths: Sequence[Path] = (),
    ) -> "AppSettings":
        if path.exists():
            return cls.load(path)
        settings = cls()
        for legacy in legacy_paths:
            if not legacy.exists():
                continue
            try:
                data = json.loads(legacy.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError):
                continue
            known = set(cls.__dataclass_fields__)
            settings = cls(**{key: value for key, value in data.items() if key in known})
            _migrate_legacy_session(legacy, data, path.parent)
            break
        settings.save(path)
        return settings


# ---------------------------------------------------------------------------
# Telegram login
# ---------------------------------------------------------------------------
async def ensure_login(
    config: AssistantConfig,
    ask_code: Callable[[], "Any"],
    ask_password: Callable[[], "Any"],
    report: Callable[[str], None] | None = None,
) -> None:
    """Make sure the session is authorised, asking for the code in the GUI."""
    from telethon import TelegramClient
    from telethon.errors import (
        PhoneCodeExpiredError,
        PhoneCodeInvalidError,
        SessionPasswordNeededError,
    )

    client = TelegramClient(config.session, config.api_id, config.api_hash)
    await client.connect()
    try:
        if await client.is_user_authorized():
            return
        if not config.phone:
            raise RuntimeError("Не указан номер телефона для входа в Telegram")
        sent = await client.send_code_request(config.phone)
        for attempt in range(3):
            code = (await ask_code() or "").strip()
            if not code:
                raise LoginCancelled()
            try:
                await client.sign_in(
                    phone=config.phone,
                    code=code,
                    phone_code_hash=sent.phone_code_hash,
                )
                return
            except SessionPasswordNeededError:
                password = await ask_password() or ""
                if not password:
                    raise LoginCancelled()
                await client.sign_in(password=password)
                return
            except (PhoneCodeInvalidError, PhoneCodeExpiredError) as error:
                if attempt == 2:
                    raise
                if report is not None:
                    report(f"Код не подошёл ({error.__class__.__name__}), попробуйте ещё раз")
    finally:
        await client.disconnect()


# ---------------------------------------------------------------------------
# Assistant worker
# ---------------------------------------------------------------------------
class AssistantWorker:
    """Owns the asyncio loop that runs :class:`TelegramAssistant`."""

    def __init__(
        self,
        settings: AppSettings,
        data_dir: Path,
        events: "queue.Queue[tuple[str, Any]]",
    ):
        self.settings = settings
        self.data_dir = data_dir
        self.events = events
        self.thread: threading.Thread | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.assistant: TelegramAssistant | None = None

    # -- thread lifecycle ---------------------------------------------------
    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._run, name="assistant", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        loop, assistant = self.loop, self.assistant
        if loop is None or assistant is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(assistant.stop(), loop)
        except RuntimeError:
            pass

    # -- worker thread ------------------------------------------------------
    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self.loop = loop
        asyncio.set_event_loop(loop)
        try:
            config = self.settings.to_assistant_config(self.data_dir)
            loop.run_until_complete(self._main(config))
        except LoginCancelled:
            self.events.put(("status", ("Вход в Telegram отменён", WARN)))
        except Exception as error:  # noqa: BLE001
            self.events.put(("log", f"ОШИБКА: {error!r}"))
            self.events.put(("log", traceback.format_exc()))
            self.events.put(("status", (f"Ошибка: {error}", ERROR)))
        finally:
            self.loop = None
            self.assistant = None
            try:
                loop.close()
            except Exception:  # noqa: BLE001
                pass
            self.events.put(("stopped", None))

    async def _main(self, config: AssistantConfig) -> None:
        from .rl import load_rl

        self.events.put(("status", ("Загружаю модель…", WARN)))
        net = load_rl(config.model_path)
        self.events.put(("status", ("Вход в Telegram…", WARN)))
        await ensure_login(
            config,
            self._ask("ask_code"),
            self._ask("ask_password"),
            report=lambda line: self.events.put(("log", line)),
        )
        assistant = TelegramAssistant(config, net)
        self.assistant = assistant
        self.events.put(("status", ("Работает — жду ходы", OK)))
        self.events.put(("log", "ассистент запущен"))
        await assistant.run()
        self.events.put(("status", ("Остановлен", MUTED)))

    def _ask(self, kind: str) -> Callable[[], Any]:
        async def ask() -> str:
            answer: "queue.Queue[str]" = queue.Queue(maxsize=1)
            self.events.put((kind, answer))
            return await asyncio.to_thread(answer.get)

        return ask


# ---------------------------------------------------------------------------
# Desktop window
# ---------------------------------------------------------------------------
class DesktopApp:
    def __init__(
        self,
        settings: AppSettings,
        data_dir: Path,
        settings_path: Path,
        *,
        start_minimized: bool = False,
    ):
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.settings = settings
        self.data_dir = data_dir
        self.settings_path = settings_path
        self.events: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        self.worker: AssistantWorker | None = None
        self.tray: Any = None
        self.running = False
        self._quitting = False

        self.root = tk.Tk()
        self.root.title(f"{APP_NAME} {APP_VERSION}")
        self.dpi_scale = max(1.0, float(self.root.winfo_fpixels("1i")) / 96.0)
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        width = min(self.px(760), screen_w - self.px(40))
        height = min(self.px(745), screen_h - self.px(130))
        x = max(0, (screen_w - width) // 2)
        y = max(0, (screen_h - height) // 2 - self.px(10))
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.minsize(
            min(self.px(640), screen_w - self.px(40)),
            min(self.px(620), screen_h - self.px(150)),
        )
        self.root.configure(bg=BG)
        self._set_window_icon()

        self.log_path = data_dir / "logs" / "assistant.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = self.log_path.open("a", encoding="utf-8")

        self._build_styles()
        self._build_ui()
        self._fill_form(settings)

        set_log_sink(self._on_log)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<Control-s>", lambda _event: self._save())
        self.root.after(300, self._poll_events)

        self._start_tray()

        if start_minimized and settings.is_configured():
            self.root.withdraw()
        else:
            self.show()

        if settings.is_configured():
            self.root.after(400, self._start_assistant)
        else:
            self._set_status("Заполните данные", WARN)

        if not (start_minimized and settings.is_configured()):
            self.root.after(700, self._maybe_show_star_reminder)

    # -- styling ------------------------------------------------------------
    def px(self, value: float) -> int:
        return int(round(value * self.dpi_scale))

    def _build_styles(self) -> None:
        style = self.ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=CARD)
        style.configure("TLabel", background=BG, foreground=FG, font=("Segoe UI", 9))
        style.configure("Card.TLabel", background=CARD, foreground=FG, font=("Segoe UI", 9))
        style.configure("Muted.TLabel", background=CARD, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Head.TLabel", background=CARD, foreground=FG, font=("Segoe UI", 10, "bold"))
        style.configure(
            "TEntry",
            fieldbackground="#ffffff",
            foreground=FG,
            bordercolor=BORDER,
            lightcolor=BORDER,
            darkcolor=BORDER,
            insertcolor=FG,
            padding=self.px(7),
        )
        style.map("TEntry", bordercolor=[("focus", ACCENT)], lightcolor=[("focus", ACCENT)])
        style.configure(
            "TCheckbutton",
            background=CARD,
            foreground=FG,
            font=("Segoe UI", 9),
            indicatorcolor="#ffffff",
        )
        style.map(
            "TCheckbutton",
            background=[("active", CARD)],
            indicatorcolor=[("selected", ACCENT)],
        )
        style.configure(
            "TButton",
            background="#f1f5f9",
            foreground=FG,
            bordercolor=BORDER,
            padding=(self.px(14), self.px(8)),
            font=("Segoe UI", 9),
        )
        style.map(
            "TButton",
            background=[("active", "#e2e8f0")],
            bordercolor=[("active", "#cbd5e1")],
        )
        style.configure(
            "Accent.TButton",
            background=ACCENT,
            foreground="white",
            bordercolor=ACCENT,
            padding=(self.px(16), self.px(8)),
            font=("Segoe UI", 9, "bold"),
        )
        style.map(
            "Accent.TButton",
            background=[("active", ACCENT_HOVER), ("disabled", "#93c5fd")],
            bordercolor=[("active", ACCENT_HOVER), ("disabled", "#93c5fd")],
            foreground=[("disabled", "#f8fafc")],
        )
        style.configure(
            "Danger.TButton",
            background="#fef2f2",
            foreground=ERROR,
            bordercolor="#fecaca",
        )
        style.map("Danger.TButton", background=[("active", "#fee2e2")])
        style.configure(
            "Ghost.TButton",
            background="#111c33",
            foreground="#dbeafe",
            bordercolor="#243452",
            padding=(self.px(10), self.px(5)),
            font=("Segoe UI", 9),
        )
        style.map("Ghost.TButton", background=[("active", "#1b2b49")])
        style.configure(
            "Link.TButton",
            background=CARD,
            foreground=ACCENT,
            borderwidth=0,
            padding=(self.px(2), self.px(2)),
            font=("Segoe UI", 9, "bold"),
        )
        style.map("Link.TButton", background=[("active", CARD)], foreground=[("active", ACCENT_HOVER)])
        style.configure("TSeparator", background=BORDER)

    def _set_window_icon(self) -> None:
        try:
            from PIL import ImageTk

            self._icon_photo = ImageTk.PhotoImage(_tray_image())
            self.root.iconphoto(True, self._icon_photo)
        except Exception:  # noqa: BLE001
            pass

    # -- layout -------------------------------------------------------------
    def _build_ui(self) -> None:
        tk = self.tk
        header = tk.Frame(self.root, bg=DARK)
        header.pack(fill="x")
        top = tk.Frame(header, bg=DARK)
        top.pack(fill="x", padx=self.px(18), pady=(self.px(12), self.px(10)))

        mark = tk.Canvas(
            top, width=self.px(38), height=self.px(38), bg=DARK,
            highlightthickness=0, bd=0,
        )
        mark.pack(side="left", padx=(0, self.px(10)))
        mark.create_oval(
            self.px(3), self.px(3), self.px(35), self.px(35),
            fill=ACCENT, outline="",
        )
        mark.create_text(
            self.px(19), self.px(19), text="8", fill="white",
            font=("Segoe UI", 17, "bold"),
        )

        titles = tk.Frame(top, bg=DARK)
        titles.pack(side="left")
        tk.Label(
            titles, text=APP_NAME, bg=DARK, fg="#f8fafc",
            font=("Segoe UI", 13, "bold"), anchor="w",
        ).pack(anchor="w")
        tk.Label(
            titles,
            text=f"8ФМ Classic · lu_luna · автоход в Telegram · v{APP_VERSION}",
            bg=DARK, fg="#94a3b8", font=("Segoe UI", 8), anchor="w",
        ).pack(anchor="w")

        self.status_label = tk.Label(
            top, text="● не запущен", bg="#13203a", fg="#94a3b8",
            font=("Segoe UI", 9, "bold"), padx=self.px(10), pady=self.px(4),
        )
        self.status_label.pack(side="right")
        self.ttk.Button(
            top, text="Гайд", style="Ghost.TButton", command=self._open_guide,
        ).pack(side="right", padx=(0, self.px(8)))
        self.ttk.Button(
            top, text="★ GitHub", style="Ghost.TButton", command=self._open_repo,
        ).pack(side="right", padx=(0, self.px(8)))

        tk.Frame(self.root, bg=ACCENT, height=self.px(2)).pack(fill="x")

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=self.px(16), pady=self.px(10))

        # The settings cards scroll vertically so the app stays usable on
        # small or high-DPI screens; the journal stays pinned at the bottom.
        scroll_host = tk.Frame(body, bg=BG)
        self.canvas = tk.Canvas(scroll_host, bg=BG, highlightthickness=0, bd=0)
        vbar = self.ttk.Scrollbar(scroll_host, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vbar.set)
        vbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        form = tk.Frame(self.canvas, bg=BG)
        self._form_id = self.canvas.create_window((0, 0), window=form, anchor="nw")
        form.bind(
            "<Configure>",
            lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.bind(
            "<Configure>",
            lambda event: self.canvas.itemconfigure(self._form_id, width=event.width),
        )
        self.root.bind_all("<MouseWheel>", self._on_mousewheel)

        gap = self.px(8)
        self._build_connection_card(form).grid(
            row=0, column=0, sticky="nsew", padx=(0, gap), pady=(0, self.px(10))
        )
        self._build_game_card(form).grid(
            row=0, column=1, sticky="nsew", padx=(gap, 0), pady=(0, self.px(10))
        )
        form.columnconfigure(0, weight=1)
        form.columnconfigure(1, weight=1)

        # Reserve the bottom blocks first, then let the settings area take the
        # remaining height and scroll.
        self._build_log_card(body).pack(side="bottom", fill="x", pady=(self.px(8), 0))
        self._build_options_card(body).pack(side="bottom", fill="x")
        scroll_host.pack(side="top", fill="both", expand=True)

    def _on_mousewheel(self, event: Any) -> None:
        widget = self.root.winfo_containing(event.x_root, event.y_root)
        while widget is not None:
            if widget is getattr(self, "log_text", None):
                return
            widget = getattr(widget, "master", None)
        self.canvas.yview_scroll(int(-event.delta / 120), "units")

    def _card(self, parent: Any) -> tuple[Any, Any]:
        tk = self.tk
        outer = tk.Frame(parent, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        tk.Frame(outer, bg=ACCENT, height=self.px(3)).pack(fill="x")
        inner = tk.Frame(outer, bg=CARD)
        inner.pack(
            fill="both", expand=True, padx=self.px(14), pady=self.px(12)
        )
        return outer, inner

    def _title(self, parent: Any, text: str, subtitle: str = "", *, pack: bool = True) -> Any:
        """Accent-marked section heading used by every card."""
        tk = self.tk
        row = tk.Frame(parent, bg=CARD)
        if pack:
            row.pack(fill="x")
        tk.Frame(row, bg=ACCENT, width=self.px(4), height=self.px(18)).pack(
            side="left", padx=(0, self.px(8))
        )
        column = tk.Frame(row, bg=CARD)
        column.pack(side="left")
        tk.Label(
            column, text=text, bg=CARD, fg=FG,
            font=("Segoe UI", 10, "bold"), anchor="w",
        ).pack(anchor="w")
        if subtitle:
            tk.Label(
                column, text=subtitle, bg=CARD, fg=MUTED,
                font=("Segoe UI", 8), anchor="w",
            ).pack(anchor="w")
        return row

    def _field_pair(self, parent: Any, items: Sequence[tuple[str, Any]]) -> None:
        """Two compact fields side by side, used to keep cards short."""
        tk = self.tk
        frame = tk.Frame(parent, bg=CARD)
        frame.pack(fill="x", pady=(0, self.px(6)))
        for index, (label, variable) in enumerate(items):
            column = tk.Frame(frame, bg=CARD)
            column.grid(
                row=0, column=index, sticky="ew",
                padx=(0, self.px(6)) if index == 0 else (self.px(6), 0),
            )
            tk.Label(
                column, text=label, bg=CARD, fg=FG,
                font=("Segoe UI", 9), anchor="w",
            ).pack(fill="x")
            self.ttk.Entry(column, textvariable=variable).pack(
                fill="x", pady=(self.px(4), 0)
            )
            frame.columnconfigure(index, weight=1)

    def _field_block(self, parent: Any, label: str, variable: Any, hint: str = "") -> None:
        tk = self.tk
        frame = tk.Frame(parent, bg=CARD)
        frame.pack(fill="x", pady=(0, self.px(6)))
        tk.Label(
            frame, text=label, bg=CARD, fg=FG, font=("Segoe UI", 9), anchor="w",
        ).pack(fill="x")
        self.ttk.Entry(frame, textvariable=variable).pack(
            fill="x", pady=(self.px(4), 0)
        )
        if hint:
            tk.Label(
                frame, text=hint, bg=CARD, fg=MUTED, font=("Segoe UI", 8),
                anchor="w", justify="left",
            ).pack(fill="x", pady=(self.px(2), 0))

    def _build_connection_card(self, parent: Any) -> Any:
        tk = self.tk
        outer, inner = self._card(parent)
        self._title(
            inner, "Подключение к Telegram",
            "api_id и api_hash — с my.telegram.org",
        )
        tk.Label(
            inner,
            text="Секреты не публикуйте. Подробная инструкция — в гайде.",
            bg=CARD, fg=MUTED, font=("Segoe UI", 8), anchor="w", justify="left",
            wraplength=self.px(320),
        ).pack(fill="x", pady=(self.px(2), self.px(10)))

        self.var_api_id = tk.StringVar()
        self.var_api_hash = tk.StringVar()
        self.var_phone = tk.StringVar()
        self._field_pair(
            inner,
            [("api_id", self.var_api_id), ("api_hash", self.var_api_hash)],
        )
        self._field_block(
            inner, "Телефон для первого входа", self.var_phone,
            hint="например +375447224002",
        )
        self.ttk.Button(
            inner, text="Где взять api_id и api_hash? →",
            style="Link.TButton", command=self._open_guide,
        ).pack(anchor="w", pady=(self.px(2), 0))
        return outer

    def _build_game_card(self, parent: Any) -> Any:
        tk = self.tk
        outer, inner = self._card(parent)
        self._title(
            inner, "Игровые данные",
            "ник должен точно совпадать с именем у бота",
        )
        tk.Label(
            inner,
            text="У других игроков ник другой — поменяйте его на свой.",
            bg=CARD, fg=MUTED, font=("Segoe UI", 8), anchor="w", justify="left",
            wraplength=self.px(320),
        ).pack(fill="x", pady=(self.px(2), self.px(10)))

        self.var_player = tk.StringVar()
        self.var_chat = tk.StringVar()
        self.var_model = tk.StringVar()
        self.var_delay = tk.StringVar()
        self._field_block(inner, "Ваш ник в игре", self.var_player)
        self._field_pair(
            inner,
            [
                ("Чат с игровым ботом", self.var_chat),
                ("Пауза перед ходом, сек", self.var_delay),
            ],
        )

        model_frame = tk.Frame(inner, bg=CARD)
        model_frame.pack(fill="x", pady=(0, 8))
        tk.Label(
            model_frame, text="Модель", bg=CARD, fg=FG, font=("Segoe UI", 9), anchor="w",
        ).pack(fill="x")
        row = tk.Frame(model_frame, bg=CARD)
        row.pack(fill="x", pady=(self.px(4), 0))
        self.ttk.Entry(row, textvariable=self.var_model).pack(
            side="left", fill="x", expand=True,
        )
        self.ttk.Button(row, text="Обзор…", command=self._browse_model).pack(
            side="left", padx=(self.px(6), 0),
        )
        self.ttk.Button(
            row, text="lu_luna",
            command=lambda: self.var_model.set(DEFAULT_MODEL),
        ).pack(side="left", padx=(self.px(6), 0))
        tk.Label(
            model_frame, text="Пусто — встроенная models/lu_luna.json.",
            bg=CARD, fg=MUTED, font=("Segoe UI", 8), anchor="w",
        ).pack(fill="x", pady=(self.px(2), 0))
        return outer

    def _build_options_card(self, parent: Any) -> Any:
        tk = self.tk
        outer, inner = self._card(parent)
        self._title(
            inner, "Поведение",
            "автозапуск, трей и безопасный пробный режим",
            pack=False,
        ).grid(row=0, column=0, sticky="w", pady=(0, self.px(8)))

        self.var_dry_run = tk.BooleanVar()
        self.var_autostart = tk.BooleanVar()
        self.var_minimized = tk.BooleanVar()
        checks = tk.Frame(inner, bg=CARD)
        checks.grid(row=1, column=0, sticky="ew", pady=(0, self.px(6)))
        self.ttk.Checkbutton(
            checks, text="Пробный режим: не нажимать кнопки",
            variable=self.var_dry_run,
        ).pack(side="left", padx=(0, self.px(16)))
        self.ttk.Checkbutton(
            checks, text="Автозапуск при входе",
            variable=self.var_autostart,
        ).pack(side="left", padx=(0, self.px(16)))
        self.ttk.Checkbutton(
            checks, text="Старт в трее",
            variable=self.var_minimized,
        ).pack(side="left")

        buttons = tk.Frame(inner, bg=CARD)
        buttons.grid(row=2, column=0, sticky="ew", pady=(self.px(8), 0))
        self.btn_save = self.ttk.Button(buttons, text="Сохранить", command=self._save)
        self.btn_save.pack(side="left")
        self.btn_start = self.ttk.Button(
            buttons, text="Запустить", style="Accent.TButton", command=self._start_assistant,
        )
        self.btn_start.pack(side="left", padx=self.px(8))
        self.btn_stop = self.ttk.Button(
            buttons, text="Остановить", style="Danger.TButton",
            command=self._stop_assistant, state="disabled",
        )
        self.btn_stop.pack(side="left", padx=self.px(8))
        self.btn_hide = self.ttk.Button(buttons, text="Свернуть в трей", command=self._hide_to_tray)
        self.btn_hide.pack(side="right")

        tk.Label(
            inner,
            text=(
                "Закрытие окна сворачивает его в трей — ассистент продолжает "
                "работать. Полный выход — через меню иконки в трее."
            ),
            bg=CARD,
            fg=MUTED,
            font=("Segoe UI", 8),
            anchor="w",
            justify="left",
            wraplength=self.px(700),
        ).grid(row=3, column=0, sticky="w", pady=(self.px(6), 0))
        inner.columnconfigure(0, weight=1)
        return outer

    def _build_log_card(self, parent: Any) -> Any:
        tk = self.tk
        outer, inner = self._card(parent)
        top = tk.Frame(inner, bg=CARD)
        top.pack(fill="x")
        tk.Frame(top, bg=OK, width=self.px(8), height=self.px(8)).pack(side="left", padx=(0, self.px(6)))
        tk.Label(
            top, text="Журнал", bg=CARD, fg=FG,
            font=("Segoe UI", 10, "bold"), anchor="w",
        ).pack(side="left")
        tk.Button(
            top, text="Очистить", bg=CARD, fg=MUTED, relief="flat",
            activebackground=CARD, cursor="hand2", command=self._clear_log,
        ).pack(side="right")
        self.log_text = tk.Text(
            inner, height=4, bg=DARK, fg=LOG_FG, insertbackground=LOG_FG,
            relief="flat", font=("Consolas", 9), wrap="word", state="disabled",
        )
        scrollbar = self.ttk.Scrollbar(inner, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side="left", fill="both", expand=True, pady=(self.px(8), 0))
        scrollbar.pack(side="right", fill="y", pady=(self.px(8), 0))
        return outer

    # -- form helpers -------------------------------------------------------
    def _fill_form(self, settings: AppSettings) -> None:
        self.var_api_id.set("" if settings.api_id is None else str(settings.api_id))
        self.var_api_hash.set(settings.api_hash)
        self.var_phone.set(settings.phone)
        self.var_player.set(settings.player_name)
        self.var_chat.set(settings.chat)
        self.var_model.set(settings.model_path.strip() or DEFAULT_MODEL)
        self.var_delay.set(f"{settings.reply_delay_seconds:g}")
        self.var_dry_run.set(settings.dry_run)
        self.var_autostart.set(settings.autostart)
        self.var_minimized.set(settings.start_minimized)

    def _collect_settings(self) -> AppSettings:
        api_raw = self.var_api_id.get().strip()
        try:
            api_id = int(api_raw) if api_raw else None
        except ValueError as error:
            raise ValueError("api_id должен быть числом") from error
        try:
            delay = float(self.var_delay.get().strip() or "20")
        except ValueError as error:
            raise ValueError("Пауза должна быть числом секунд") from error
        if delay < 0:
            raise ValueError("Пауза не может быть отрицательной")
        return AppSettings(
            api_id=api_id,
            api_hash=self.var_api_hash.get().strip(),
            phone=normalize_phone(self.var_phone.get()),
            chat=self.var_chat.get().strip() or "@phys8math8bot",
            player_name=self.var_player.get().strip() or "Леончик",
            model_path=self.var_model.get().strip() or DEFAULT_MODEL,
            reply_delay_seconds=delay,
            dry_run=bool(self.var_dry_run.get()),
            autostart=bool(self.var_autostart.get()),
            start_minimized=bool(self.var_minimized.get()),
            star_reminder_shown=self.settings.star_reminder_shown,
        )

    def _browse_model(self) -> None:
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            parent=self.root,
            title="Выберите модель",
            filetypes=[("Модель 8ФМ", "*.json"), ("Все файлы", "*.*")],
        )
        if path:
            self.var_model.set(path)

    # -- guide, repo, first-run star reminder -------------------------------
    def _open_url(self, url: str) -> None:
        import webbrowser

        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001
            pass

    def _open_repo(self) -> None:
        self._open_url(REPO_URL or "https://github.com")

    def _open_guide(self) -> None:
        tk = self.tk
        window = tk.Toplevel(self.root)
        window.title("Гайд — как подключить 8ФМ Ассистент")
        self._center_window(window, 780, 650)
        window.configure(bg=CARD)
        window.transient(self.root)

        header = tk.Frame(window, bg=DARK)
        header.pack(fill="x")
        tk.Label(
            header, text="Как подключить ассистента",
            bg=DARK, fg="#f8fafc", font=("Segoe UI", 13, "bold"),
            anchor="w",
        ).pack(fill="x", padx=self.px(18), pady=(self.px(12), 0))
        tk.Label(
            header,
            text="Пошаговая инструкция: api_id, api_hash, вход, трей и типовые ошибки",
            bg=DARK, fg="#94a3b8", font=("Segoe UI", 9), anchor="w",
        ).pack(fill="x", padx=self.px(18), pady=(0, self.px(12)))
        tk.Frame(window, bg=ACCENT, height=self.px(2)).pack(fill="x")

        buttons = tk.Frame(window, bg=CARD)
        buttons.pack(side="bottom", fill="x", padx=self.px(14), pady=(0, self.px(12)))
        body = tk.Frame(window, bg=CARD)
        body.pack(fill="both", expand=True, padx=self.px(14), pady=self.px(12))
        text = tk.Text(
            body, wrap="word", bg=CARD_SOFT, fg=FG,
            relief="flat", font=("Segoe UI", 9),
            padx=self.px(14), pady=self.px(12), spacing1=2, spacing3=4,
            insertbackground=FG,
        )
        bar = self.ttk.Scrollbar(body, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        text.pack(side="left", fill="both", expand=True)
        text.insert("1.0", GUIDE_TEXT)
        text.configure(state="disabled")
        text.tag_configure("link", foreground=ACCENT, underline=True)

        self.ttk.Button(
            buttons, text="Открыть my.telegram.org", style="Accent.TButton",
            command=lambda: self._open_url("https://my.telegram.org"),
        ).pack(side="left")
        self.ttk.Button(
            buttons, text="Открыть Telegram", command=lambda: self._open_url("https://web.telegram.org"),
        ).pack(side="left", padx=self.px(8))
        self.ttk.Button(
            buttons, text="Закрыть", command=window.destroy,
        ).pack(side="right")

    def _center_window(self, window: Any, width: int, height: int) -> None:
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        w = min(width, screen_w - self.px(40))
        h = min(height, screen_h - self.px(120))
        x = max(0, (screen_w - w) // 2)
        y = max(0, (screen_h - h) // 2 - self.px(10))
        window.geometry(f"{w}x{h}+{x}+{y}")

    def _maybe_show_star_reminder(self) -> None:
        if self.settings.star_reminder_shown:
            return
        self.settings.star_reminder_shown = True
        try:
            self.settings.save(self.settings_path)
        except OSError:
            pass
        self._show_star_reminder()

    def _show_star_reminder(self) -> None:
        tk = self.tk
        window = tk.Toplevel(self.root)
        window.title("Спасибо!")
        self._center_window(window, 470, 310)
        window.configure(bg=CARD)
        window.transient(self.root)
        window.resizable(False, False)

        hero = tk.Frame(window, bg=DARK)
        hero.pack(fill="x")
        tk.Label(
            hero, text="★", bg=DARK, fg="#fbbf24",
            font=("Segoe UI", 30, "bold"),
        ).pack(pady=(self.px(12), 0))
        tk.Label(
            hero, text="Проект живёт благодаря звёздам",
            bg=DARK, fg="#f8fafc", font=("Segoe UI", 13, "bold"),
        ).pack(pady=(0, self.px(4)))
        tk.Label(
            hero, text="8ФМ Ассистент — бесплатный и открытый",
            bg=DARK, fg="#94a3b8", font=("Segoe UI", 9),
        ).pack(pady=(0, self.px(14)))
        tk.Frame(window, bg="#fbbf24", height=self.px(2)).pack(fill="x")

        tk.Label(
            window,
            text=(
                "Если приложение помогает тебе за столом,\n"
                "поставь звезду на GitHub — это лучшая\n"
                "поддержка и повод развивать проект дальше."
            ),
            bg=CARD, fg=FG, font=("Segoe UI", 9), justify="center",
        ).pack(fill="x", padx=self.px(18), pady=(self.px(16), self.px(14)))

        buttons = tk.Frame(window, bg=CARD)
        buttons.pack(fill="x", padx=self.px(18), pady=(0, self.px(16)))
        self.ttk.Button(
            buttons, text="★ Поставить звезду", style="Accent.TButton",
            command=lambda: (self._open_repo(), window.destroy()),
        ).pack(side="left")
        self.ttk.Button(
            buttons, text="Позже", command=window.destroy,
        ).pack(side="right")

    # -- actions ------------------------------------------------------------
    def _save(self) -> bool:
        from tkinter import messagebox

        try:
            settings = self._collect_settings()
        except ValueError as error:
            messagebox.showerror(APP_NAME, str(error), parent=self.root)
            return False
        settings.save(self.settings_path)
        self.settings = settings
        try:
            autostart.set_enabled(settings.autostart)
        except Exception as error:  # noqa: BLE001
            messagebox.showwarning(
                APP_NAME,
                f"Не удалось изменить автозапуск: {error}",
                parent=self.root,
            )
        self._set_status("Настройки сохранены", OK)
        self._append_log("настройки сохранены")
        return True

    def _start_assistant(self) -> None:
        if self.running:
            return
        if not self._save():
            return
        if not self.settings.is_configured():
            from tkinter import messagebox

            messagebox.showwarning(
                APP_NAME,
                "Укажите api_id, api_hash и телефон — без них вход в Telegram невозможен.",
                parent=self.root,
            )
            return
        self.worker = AssistantWorker(self.settings, self.data_dir, self.events)
        self.running = True
        self._update_buttons()
        self._set_status("Запускаю…", WARN)
        self.worker.start()

    def _stop_assistant(self) -> None:
        if not self.running or self.worker is None:
            return
        self._set_status("Останавливаю…", WARN)
        self.worker.stop()

    def _clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    # -- tray ---------------------------------------------------------------
    def _start_tray(self) -> None:
        try:
            import pystray

            menu = pystray.Menu(
                pystray.MenuItem(
                    "Открыть окно",
                    lambda: self.events.put(("open", None)),
                    default=True,
                ),
                pystray.MenuItem(
                    lambda _item: "Остановить ассистент" if self.running else "Запустить ассистент",
                    lambda: self.events.put(("toggle", None)),
                ),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Выход", lambda: self.events.put(("quit", None))),
            )
            self.tray = pystray.Icon("8fm-assistant", _tray_image(), APP_NAME, menu)
            threading.Thread(target=self.tray.run, name="tray", daemon=True).start()
            self._append_log(f"трей запущен, журнал: {self.log_path}")
        except Exception as error:  # noqa: BLE001
            self.tray = None
            self._append_log(f"трей недоступен: {error!r}")

    def _hide_to_tray(self) -> None:
        if self.tray is not None:
            self.root.withdraw()
            try:
                self.tray.notify("Ассистент продолжает работать в трее", APP_NAME)
            except Exception:  # noqa: BLE001
                pass
        else:
            self.root.iconify()

    def _on_close(self) -> None:
        if self.tray is not None:
            self._hide_to_tray()
        else:
            self.quit()

    # -- event pump ---------------------------------------------------------
    def _on_log(self, line: str) -> None:
        self.events.put(("log", line))

    def _poll_events(self) -> None:
        handled = 0
        try:
            while True:
                kind, payload = self.events.get_nowait()
                handled += 1
                if kind == "log":
                    self._append_log(str(payload))
                elif kind == "status":
                    self._set_status(*payload)
                elif kind == "ask_code":
                    self._answer_login("ask_code", payload)
                elif kind == "ask_password":
                    self._answer_login("ask_password", payload)
                elif kind == "open":
                    self.show()
                elif kind == "toggle":
                    if self.running:
                        self._stop_assistant()
                    else:
                        self._start_assistant()
                elif kind == "quit":
                    self.quit()
                    return
                elif kind == "stopped":
                    self.running = False
                    self._update_buttons()
        except queue.Empty:
            pass
        if not self._quitting:
            # Idle in the tray: two wake-ups per second instead of ten.  After
            # real activity answer faster so the UI still feels instant.
            try:
                hidden = self.root.state() == "withdrawn"
            except Exception:  # noqa: BLE001
                hidden = False
            if handled:
                delay = 120
            elif hidden:
                delay = 750
            else:
                delay = 500
            self.root.after(delay, self._poll_events)

    def _answer_login(self, kind: str, answer: "queue.Queue[str]") -> None:
        from tkinter import simpledialog

        self.show()
        if kind == "ask_code":
            value = simpledialog.askstring(
                "Вход в Telegram",
                "Telegram прислал код в чат от Telegram.\nВведите код:",
                parent=self.root,
            )
        else:
            value = simpledialog.askstring(
                "Пароль 2FA",
                "Введите пароль двухэтапной проверки Telegram:",
                parent=self.root,
                show="*",
            )
        answer.put(value if value is not None else "")

    def _append_log(self, line: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
        try:
            self.log_handle.write(line + "\n")
            self.log_handle.flush()
        except OSError:
            pass

    def _set_status(self, text: str, color: str = MUTED) -> None:
        short = text if len(text) <= 48 else text[:47] + "…"
        self.status_label.configure(text=f"● {short}", fg=color)
        if self.tray is not None:
            try:
                self.tray.title = f"{APP_NAME} — {text}"
            except Exception:  # noqa: BLE001
                pass

    def _update_buttons(self) -> None:
        self.btn_start.configure(state="disabled" if self.running else "normal")
        self.btn_stop.configure(state="normal" if self.running else "disabled")

    # -- window -------------------------------------------------------------
    def show(self) -> None:
        try:
            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()
        except Exception:  # noqa: BLE001
            pass

    def quit(self) -> None:
        if self._quitting:
            return
        self._quitting = True
        set_log_sink(None)
        if self.worker is not None:
            self.worker.stop()
        if self.tray is not None:
            try:
                self.tray.stop()
            except Exception:  # noqa: BLE001
                pass
            self.tray = None
        try:
            self.log_handle.close()
        except Exception:  # noqa: BLE001
            pass
        self.root.after(150, self.root.destroy)

    def run(self) -> None:
        self.root.mainloop()


# ---------------------------------------------------------------------------
# Tray icon image
# ---------------------------------------------------------------------------
def _tray_image() -> Any:
    from PIL import Image, ImageDraw, ImageFont

    size = 64
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (2, 2, size - 3, size - 3),
        radius=15,
        fill=(17, 24, 39, 255),
        outline=(37, 99, 235, 255),
        width=3,
    )
    font = None
    for name in ("segoeuib.ttf", "arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            font = ImageFont.truetype(name, 34)
            break
        except OSError:
            continue
    text = "8"
    kwargs: dict[str, Any] = {}
    if font is not None:
        kwargs["font"] = font
    try:
        left, top, right, bottom = draw.textbbox((0, 0), text, **kwargs)
    except Exception:  # noqa: BLE001
        left, top, right, bottom = 0, 0, 0, 0
    x = (size - (right - left)) / 2 - left
    y = (size - (bottom - top)) / 2 - top
    draw.text((x, y), text, fill=(255, 255, 255, 255), **kwargs)
    return image


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _redirect_std_streams(data_dir: Path) -> None:
    """``pythonw.exe`` gives ``sys.stdout``/``stderr`` as ``None``."""
    log_path = data_dir / "logs" / "console.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if sys.stdout is None:
        sys.stdout = open(log_path, "a", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(log_path, "a", encoding="utf-8")


def _acquire_single_instance() -> Any:
    if os.name != "nt":
        return object()
    import ctypes
    import ctypes.wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.restype = ctypes.wintypes.HANDLE
    kernel32.CreateMutexW.argtypes = [
        ctypes.wintypes.LPVOID,
        ctypes.wintypes.BOOL,
        ctypes.wintypes.LPCWSTR,
    ]
    kernel32.CloseHandle.restype = ctypes.wintypes.BOOL
    kernel32.CloseHandle.argtypes = [ctypes.wintypes.HANDLE]
    handle = kernel32.CreateMutexW(None, False, "8FM.Assistant.SingleInstance")
    if not handle:
        return None
    if ctypes.get_last_error() == 183:  # ERROR_ALREADY_EXISTS
        kernel32.CloseHandle(handle)
        return None
    return handle


def _warn_already_running() -> None:
    if os.name == "nt":
        import ctypes
        import ctypes.wintypes

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.MessageBoxW.restype = ctypes.c_int
        user32.MessageBoxW.argtypes = [
            ctypes.wintypes.HWND,
            ctypes.wintypes.LPCWSTR,
            ctypes.wintypes.LPCWSTR,
            ctypes.wintypes.UINT,
        ]
        user32.MessageBoxW(
            None,
            "8ФМ Ассистент уже запущен. Ищите иконку в системном трее.",
            APP_NAME,
            0x40,
        )
    else:
        print("8ФМ Ассистент уже запущен.", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="8ФМ Ассистент — окно и трей")
    parser.add_argument("--minimized", action="store_true", help="Start hidden in the tray")
    parser.add_argument("--config", help="Path to the settings JSON")
    args = parser.parse_args(list(argv) if argv is not None else None)

    data_dir = user_data_dir()
    _redirect_std_streams(data_dir)

    if os.name == "nt":
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:  # noqa: BLE001
            pass

    settings_path = Path(args.config).expanduser() if args.config else data_dir / "config.json"
    settings = AppSettings.load_or_seed(settings_path, legacy_config_paths())

    if settings.autostart:
        # Refresh the registered command on every start: a frozen build should
        # point autostart at the .exe, a source run at pythonw + launcher.
        try:
            autostart.set_enabled(True)
        except Exception:  # noqa: BLE001
            pass

    mutex = _acquire_single_instance()
    if mutex is None:
        _warn_already_running()
        return 1

    app = DesktopApp(
        settings,
        data_dir,
        settings_path,
        start_minimized=bool(args.minimized or settings.start_minimized),
    )
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
