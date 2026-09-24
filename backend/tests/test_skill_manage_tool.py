import asyncio
import importlib
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anyio
import pytest

from deerflow.skills.security_static_scanner import StaticScannerError

skill_manage_module = importlib.import_module("deerflow.tools.skill_manage_tool")


def _skill_content(name: str, description: str = "Demo skill") -> str:
    return f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n"


async def _async_result(decision: str, reason: str):
    from deerflow.skills.security_scanner import ScanResult

    return ScanResult(decision=decision, reason=reason)


def _make_config(skills_root: Path):
    return SimpleNamespace(
        skills=SimpleNamespace(
            get_skills_path=lambda: skills_root,
            container_path="/mnt/skills",
            use="deerflow.skills.storage.local_skill_storage:LocalSkillStorage",
        ),
        skill_evolution=SimpleNamespace(enabled=True, moderation_model_name=None),
    )


def _make_runtime(*, thread_id: str = "thread-1", user_id: str = "default"):
    return SimpleNamespace(
        context={"thread_id": thread_id, "user_id": user_id},
        config={"configurable": {"thread_id": thread_id, "user_id": user_id}},
    )


def test_skill_manage_create_and_patch(monkeypatch, tmp_path):
    skills_root = tmp_path / "skills"
    config = _make_config(skills_root)
    monkeypatch.setattr("deerflow.config.get_app_config", lambda: config)
    monkeypatch.setattr("deerflow.skills.security_scanner.get_app_config", lambda: config)
    # Patch get_paths so UserScopedSkillStorage resolves user dirs under tmp_path
    from deerflow.config.paths import Paths

    monkeypatch.setattr("deerflow.config.paths.get_paths", lambda: Paths(base_dir=tmp_path))
    monkeypatch.setattr("deerflow.config.paths._paths", None)

    refresh_calls = []

    async def _refresh(user_id: str):
        refresh_calls.append(("refresh", user_id))

    monkeypatch.setattr(skill_manage_module, "refresh_user_skills_system_prompt_cache_async", _refresh)
    monkeypatch.setattr(skill_manage_module, "scan_skill_content", lambda *args, **kwargs: _async_result("allow", "ok"))

    runtime = _make_runtime(user_id="default")

    result = anyio.run(
        skill_manage_module.skill_manage_tool.coroutine,
        runtime,
        "create",
        "demo-skill",
        _skill_content("demo-skill"),
    )
    assert "Created custom skill" in result

    patch_result = anyio.run(
        skill_manage_module.skill_manage_tool.coroutine,
        runtime,
        "patch",
        "demo-skill",
        None,
        None,
        "Demo skill",
        "Patched skill",
        1,
    )
    assert "Patched custom skill" in patch_result
    # User-scoped: custom skills written under users/default/skills/custom/
    user_custom = tmp_path / "users" / "default" / "skills" / "custom"
    assert "Patched skill" in (user_custom / "demo-skill" / "SKILL.md").read_text(encoding="utf-8")
    assert refresh_calls == [("refresh", "default"), ("refresh", "default")]


def test_skill_manage_patch_replaces_single_occurrence_by_default(monkeypatch, tmp_path):
    skills_root = tmp_path / "skills"
    config = _make_config(skills_root)
    monkeypatch.setattr("deerflow.config.get_app_config", lambda: config)
    monkeypatch.setattr("deerflow.skills.security_scanner.get_app_config", lambda: config)
    from deerflow.config.paths import Paths

    monkeypatch.setattr("deerflow.config.paths.get_paths", lambda: Paths(base_dir=tmp_path))
    monkeypatch.setattr("deerflow.config.paths._paths", None)

    async def _refresh(user_id: str):
        return None

    monkeypatch.setattr(skill_manage_module, "refresh_user_skills_system_prompt_cache_async", _refresh)
    monkeypatch.setattr(skill_manage_module, "scan_skill_content", lambda *args, **kwargs: _async_result("allow", "ok"))

    runtime = _make_runtime(user_id="default")
    content = _skill_content("demo-skill", "Demo skill") + "\nRepeated: Demo skill\n"

    anyio.run(skill_manage_module.skill_manage_tool.coroutine, runtime, "create", "demo-skill", content)
    patch_result = anyio.run(
        skill_manage_module.skill_manage_tool.coroutine,
        runtime,
        "patch",
        "demo-skill",
        None,
        None,
        "Demo skill",
        "Patched skill",
    )

    user_custom = tmp_path / "users" / "default" / "skills" / "custom"
    skill_text = (user_custom / "demo-skill" / "SKILL.md").read_text(encoding="utf-8")
    assert "1 replacement(s) applied, 2 match(es) found" in patch_result
    assert skill_text.count("Patched skill") == 1
    assert skill_text.count("Demo skill") == 1


def test_skill_manage_rejects_public_skill_patch(monkeypatch, tmp_path):
    skills_root = tmp_path / "skills"
    public_dir = skills_root / "public" / "deep-research"
    public_dir.mkdir(parents=True, exist_ok=True)
    (public_dir / "SKILL.md").write_text(_skill_content("deep-research"), encoding="utf-8")
    config = _make_config(skills_root)
    monkeypatch.setattr("deerflow.config.get_app_config", lambda: config)
    from deerflow.config.paths import Paths

    monkeypatch.setattr("deerflow.config.paths.get_paths", lambda: Paths(base_dir=tmp_path))
    monkeypatch.setattr("deerflow.config.paths._paths", None)

    runtime = _make_runtime(user_id="default")

    with pytest.raises(ValueError, match="built-in skill"):
        anyio.run(
            skill_manage_module.skill_manage_tool.coroutine,
            runtime,
            "patch",
            "deep-research",
            None,
            None,
            "Demo skill",
            "Patched",
        )


def test_skill_manage_sync_wrapper_supported(monkeypatch, tmp_path):
    skills_root = tmp_path / "skills"
    config = _make_config(skills_root)
    monkeypatch.setattr("deerflow.config.get_app_config", lambda: config)
    from deerflow.config.paths import Paths

    monkeypatch.setattr("deerflow.config.paths.get_paths", lambda: Paths(base_dir=tmp_path))
    monkeypatch.setattr("deerflow.config.paths._paths", None)

    refresh_calls = []

    async def _refresh(user_id: str):
        refresh_calls.append(("refresh", user_id))

    monkeypatch.setattr(skill_manage_module, "refresh_user_skills_system_prompt_cache_async", _refresh)
    monkeypatch.setattr(skill_manage_module, "scan_skill_content", lambda *args, **kwargs: _async_result("allow", "ok"))

    runtime = _make_runtime(thread_id="thread-sync", user_id="default")
    result = skill_manage_module.skill_manage_tool.func(
        runtime=runtime,
        action="create",
        name="sync-skill",
        content=_skill_content("sync-skill"),
    )

    assert "Created custom skill" in result
    assert refresh_calls == [("refresh", "default")]


def test_skill_manage_rejects_support_path_traversal(monkeypatch, tmp_path):
    skills_root = tmp_path / "skills"
    config = _make_config(skills_root)
    monkeypatch.setattr("deerflow.config.get_app_config", lambda: config)
    monkeypatch.setattr("deerflow.skills.security_scanner.get_app_config", lambda: config)
    from deerflow.config.paths import Paths

    monkeypatch.setattr("deerflow.config.paths.get_paths", lambda: Paths(base_dir=tmp_path))
    monkeypatch.setattr("deerflow.config.paths._paths", None)

    async def _refresh(user_id: str):
        return None

    monkeypatch.setattr(skill_manage_module, "refresh_user_skills_system_prompt_cache_async", _refresh)
    monkeypatch.setattr(skill_manage_module, "scan_skill_content", lambda *args, **kwargs: _async_result("allow", "ok"))

    runtime = _make_runtime(user_id="default")
    anyio.run(skill_manage_module.skill_manage_tool.coroutine, runtime, "create", "demo-skill", _skill_content("demo-skill"))

    with pytest.raises(ValueError, match="parent-directory traversal|selected support directory"):
        anyio.run(
            skill_manage_module.skill_manage_tool.coroutine,
            runtime,
            "write_file",
            "demo-skill",
            "malicious overwrite",
            "references/../SKILL.md",
        )


def test_skill_manage_remove_file_updates_sandbox_projection_before_return(monkeypatch, tmp_path):
    skills_root = tmp_path / "skills"
    config = _make_config(skills_root)
    monkeypatch.setattr("deerflow.config.get_app_config", lambda: config)
    monkeypatch.setattr("deerflow.skills.security_scanner.get_app_config", lambda: config)
    from deerflow.config.paths import Paths

    monkeypatch.setattr("deerflow.config.paths.get_paths", lambda: Paths(base_dir=tmp_path))
    monkeypatch.setattr("deerflow.config.paths._paths", None)

    async def _refresh(user_id: str):
        return None

    monkeypatch.setattr(skill_manage_module, "refresh_user_skills_system_prompt_cache_async", _refresh)
    monkeypatch.setattr(skill_manage_module, "scan_skill_content", lambda *args, **kwargs: _async_result("allow", "ok"))

    runtime = _make_runtime(user_id="default")
    anyio.run(skill_manage_module.skill_manage_tool.coroutine, runtime, "create", "demo-skill", _skill_content("demo-skill"))
    anyio.run(
        skill_manage_module.skill_manage_tool.coroutine,
        runtime,
        "write_file",
        "demo-skill",
        "supporting content",
        "references/guide.md",
    )
    projected_file = tmp_path / "users" / "default" / "skills_view" / "custom" / "demo-skill" / "references" / "guide.md"
    assert projected_file.read_text(encoding="utf-8") == "supporting content"

    result = anyio.run(
        skill_manage_module.skill_manage_tool.coroutine,
        runtime,
        "remove_file",
        "demo-skill",
        None,
        "references/guide.md",
    )

    assert result == "Removed 'references/guide.md' from custom skill 'demo-skill'."
    assert not projected_file.exists()


def test_skill_manage_static_critical_blocks_create_before_llm(monkeypatch, tmp_path):
    skills_root = tmp_path / "skills"
    config = _make_config(skills_root)
    monkeypatch.setattr("deerflow.config.get_app_config", lambda: config)
    monkeypatch.setattr("deerflow.skills.security_scanner.get_app_config", lambda: config)
    from deerflow.config.paths import Paths

    monkeypatch.setattr("deerflow.config.paths.get_paths", lambda: Paths(base_dir=tmp_path))
    monkeypatch.setattr("deerflow.config.paths._paths", None)
    refresh_calls = []
    llm_calls = []

    async def _refresh(user_id: str):
        refresh_calls.append(("refresh", user_id))

    async def _scan(*args, **kwargs):
        llm_calls.append({"args": args, "kwargs": kwargs})
        return await _async_result("allow", "ok")

    monkeypatch.setattr(skill_manage_module, "refresh_user_skills_system_prompt_cache_async", _refresh)
    monkeypatch.setattr(skill_manage_module, "scan_skill_content", _scan)

    runtime = _make_runtime(user_id="default")
    content = _skill_content("blocked-skill") + "\n-----BEGIN RSA PRIVATE KEY-----\nabc\n-----END RSA PRIVATE KEY-----\n"

    with pytest.raises(ValueError) as excinfo:
        anyio.run(
            skill_manage_module.skill_manage_tool.coroutine,
            runtime,
            "create",
            "blocked-skill",
            content,
        )

    assert "Static security scan blocked" in str(excinfo.value)
    assert "secret-private-key" in str(excinfo.value)
    assert llm_calls == []
    assert refresh_calls == []
    assert not (tmp_path / "users" / "default" / "skills" / "custom" / "blocked-skill" / "SKILL.md").exists()


def test_skill_manage_static_scan_failure_blocks_create_before_llm(monkeypatch, tmp_path):
    skills_root = tmp_path / "skills"
    config = _make_config(skills_root)
    monkeypatch.setattr("deerflow.config.get_app_config", lambda: config)
    monkeypatch.setattr("deerflow.skills.security_scanner.get_app_config", lambda: config)
    from deerflow.config.paths import Paths

    monkeypatch.setattr("deerflow.config.paths.get_paths", lambda: Paths(base_dir=tmp_path))
    monkeypatch.setattr("deerflow.config.paths._paths", None)
    refresh_calls = []
    llm_calls = []

    async def _refresh(user_id: str):
        refresh_calls.append(("refresh", user_id))

    async def _scan(*args, **kwargs):
        llm_calls.append({"args": args, "kwargs": kwargs})
        return await _async_result("allow", "ok")

    def _broken_static_scan(skill_dir, *, skill_name=None, app_config=None):
        raise StaticScannerError("native scanner unavailable")

    monkeypatch.setattr(skill_manage_module, "refresh_user_skills_system_prompt_cache_async", _refresh)
    monkeypatch.setattr(skill_manage_module, "scan_skill_content", _scan)
    monkeypatch.setattr(skill_manage_module, "enforce_static_scan", _broken_static_scan)

    runtime = _make_runtime(user_id="default")

    with pytest.raises(ValueError, match="Static security scan failed.*native scanner unavailable"):
        anyio.run(
            skill_manage_module.skill_manage_tool.coroutine,
            runtime,
            "create",
            "scanner-failure-skill",
            _skill_content("scanner-failure-skill"),
        )

    assert llm_calls == []
    assert refresh_calls == []
    assert not (tmp_path / "users" / "default" / "skills" / "custom" / "scanner-failure-skill" / "SKILL.md").exists()


def test_skill_manage_per_user_isolation(monkeypatch, tmp_path):
    """Two different users must get separate custom skill directories."""
    skills_root = tmp_path / "skills"
    config = _make_config(skills_root)
    monkeypatch.setattr("deerflow.config.get_app_config", lambda: config)
    monkeypatch.setattr("deerflow.skills.security_scanner.get_app_config", lambda: config)
    from deerflow.config.paths import Paths

    monkeypatch.setattr("deerflow.config.paths.get_paths", lambda: Paths(base_dir=tmp_path))
    monkeypatch.setattr("deerflow.config.paths._paths", None)

    async def _refresh(user_id: str):
        return None

    monkeypatch.setattr(skill_manage_module, "refresh_user_skills_system_prompt_cache_async", _refresh)
    monkeypatch.setattr(skill_manage_module, "scan_skill_content", lambda *args, **kwargs: _async_result("allow", "ok"))

    # Alice creates a skill
    runtime_alice = _make_runtime(user_id="alice")
    result_a = anyio.run(
        skill_manage_module.skill_manage_tool.coroutine,
        runtime_alice,
        "create",
        "alice-skill",
        _skill_content("alice-skill"),
    )
    assert "Created custom skill" in result_a

    # Bob creates a different skill
    runtime_bob = _make_runtime(user_id="bob")
    result_b = anyio.run(
        skill_manage_module.skill_manage_tool.coroutine,
        runtime_bob,
        "create",
        "bob-skill",
        _skill_content("bob-skill"),
    )
    assert "Created custom skill" in result_b

    # Verify separate directories
    alice_dir = tmp_path / "users" / "alice" / "skills" / "custom" / "alice-skill"
    bob_dir = tmp_path / "users" / "bob" / "skills" / "custom" / "bob-skill"
    assert alice_dir.exists()
    assert bob_dir.exists()
    # No cross-contamination
    assert not (tmp_path / "users" / "alice" / "skills" / "custom" / "bob-skill").exists()
    assert not (tmp_path / "users" / "bob" / "skills" / "custom" / "alice-skill").exists()


# --- tracing wiring: the in-graph choke point (see the INVARIANT in
# packages/harness/deerflow/agents/lead_agent/agent.py) ---


def test_skill_manage_concurrent_sync_calls_same_user_skill_no_deadlock(monkeypatch):
    """Two skill_manage calls for the same (user, skill) in one turn must not hang.

    The embedded/TUI sync tool-call path (``DeerFlowClient.stream()`` ->
    LangGraph's ``ToolNode._func`` -> a ``ThreadPoolExecutor`` ->
    ``deerflow.tools.sync.make_sync_tool_wrapper``'s per-call ``asyncio.run()``)
    runs each of the turn's tool calls on a fresh event loop on a fresh OS
    thread, so one turn may issue two ``skill_manage`` calls for the same
    (user, skill) that contend on the one ``_get_lock(user_id, name)`` lock
    from two different event loops. A per-(user, skill) ``asyncio.Lock`` binds
    to whichever loop first contends on it; when the holder on the other loop
    later releases, the waiter's wake-up is delivered without
    ``call_soon_threadsafe``, so the waiting loop's selector is never woken
    and that call hangs forever with no exception -- and the ``executor.map``
    driving the turn waits for it, so the whole tool step never returns.

    This test uses a bounded thread-join timeout so that a regression back to
    ``asyncio.Lock`` fails this test quickly instead of hanging the whole
    suite.
    """
    entered_critical_section = threading.Event()

    class _SlowStorage:
        def __init__(self):
            self.entries = 0
            self.max_concurrent_entries = 0

        def public_skill_exists(self, name):
            # Signal that this call is inside the critical section (the lock is
            # held) and stay there briefly so the other thread has time to
            # reach its own acquire() and genuinely contend, rather than racing
            # to also take an uncontended fast path.
            self.entries += 1
            self.max_concurrent_entries = max(self.max_concurrent_entries, self.entries)
            entered_critical_section.set()
            time.sleep(0.3)
            self.entries -= 1
            return False

    storage = _SlowStorage()
    monkeypatch.setattr(skill_manage_module, "get_or_new_user_skill_storage", lambda user_id: storage)

    results: dict[str, Any] = {}

    def call(tag: str, wait_for_holder: bool) -> None:
        if wait_for_holder:
            # Only start once another thread is confirmed to be holding the
            # lock, guaranteeing this call contends instead of racing for the
            # uncontended fast path itself.
            assert entered_critical_section.wait(timeout=5), "holder thread never entered critical section"
        try:
            results[tag] = skill_manage_module.skill_manage_tool.func(
                runtime=_make_runtime(thread_id="thread-concurrent", user_id="default"),
                action="bogus",
                name="concurrent-skill",
            )
        except BaseException as exc:  # noqa: BLE001 - captured to assert below
            results[tag] = exc

    threads = [
        threading.Thread(target=call, args=("holder", False), name="holder", daemon=True),
        threading.Thread(target=call, args=("waiter", True), name="waiter", daemon=True),
    ]

    for t in threads:
        t.start()

    # Bounded timeout: under the old per-(user, skill) asyncio.Lock, the
    # waiter thread would never return. Joining with a timeout keeps a
    # regression from hanging the test suite forever; it fails fast instead.
    for t in threads:
        t.join(timeout=5)

    still_alive = [t.name for t in threads if t.is_alive()]
    assert not still_alive, f"deadlock: thread(s) still blocked after bounded timeout: {still_alive}"

    # Both calls must have run the full lock-protected section: the bogus
    # action falls through to the trailing public-skill check inside the lock
    # and raises the unsupported-action error.
    assert set(results) == {"holder", "waiter"}, f"calls did not both complete: {results!r}"
    for tag, result in results.items():
        assert isinstance(result, ValueError), f"{tag} did not hit the expected unsupported-action error: {result!r}"
        assert "Unsupported action 'bogus'" in str(result)

    # Mutual exclusion must be preserved: the waiter may only enter the
    # critical section after the holder's stay inside it finishes, so the
    # storage lookup is never entered by both calls at once.
    assert storage.max_concurrent_entries == 1


def test_skill_manage_cancelled_while_waiting_does_not_leak_lock(monkeypatch):
    """A caller cancelled while waiting on the per-(user, skill) lock must not leak it.

    ``_skill_manage_impl`` runs ``lock.acquire()`` on a real OS thread via
    ``asyncio.to_thread`` so a blocking wait never blocks the event loop. Once
    that thread has actually started running ``lock.acquire()``, Python cannot
    interrupt it: cancelling the *caller* only stops the caller from
    continuing, it does not stop the thread. If cancellation at that await let
    the thread go on to acquire the lock unobserved (nobody left holding a
    reference that will call ``release()`` for it), the lock would stay held
    forever and every subsequent call for this (user, skill) would block
    permanently at the same line -- a different path to the same permanent
    hang as the cross-loop deadlock above.

    This test holds the per-(user, skill) lock (simulating another in-flight
    call), starts a second call that has to wait for it, cancels that waiter
    while it is genuinely blocked in its executor thread, releases the
    original holder, and then asserts a third call completes within a bounded
    timeout. Every potentially-hanging await is wrapped in a bounded timeout
    so a regression fails this test quickly instead of hanging the suite.
    """
    entered = []

    class _StubStorage:
        def public_skill_exists(self, name):
            entered.append(name)
            return False

    storage = _StubStorage()
    monkeypatch.setattr(skill_manage_module, "get_or_new_user_skill_storage", lambda user_id: storage)

    runtime = _make_runtime(thread_id="thread-cancel", user_id="default")
    lock = skill_manage_module._get_lock("default", "cancel-skill")

    async def scenario() -> None:
        # Simulate another in-flight call already holding the per-(user, skill)
        # lock (uncontended, so this succeeds immediately without blocking).
        lock.acquire()
        try:
            waiter = asyncio.create_task(skill_manage_module.skill_manage_tool.coroutine(runtime, "bogus", "cancel-skill"))

            # Let the waiter's asyncio.to_thread(lock.acquire) actually get
            # scheduled onto an executor thread and start genuinely blocking
            # on the real lock before cancelling it -- otherwise the
            # cancellation could land before the thread even starts, which
            # would not exercise the leak.
            await asyncio.sleep(0.2)

            waiter.cancel()
            # The original holder finishes its own work and releases *before*
            # we wait on the cancelled waiter: a correct fix must keep the
            # lock's eventual acquisition shielded from this coroutine's
            # cancellation and wait for it to actually land before releasing,
            # so awaiting the cancelled waiter can legitimately block until
            # the lock is free either way.
            lock.release()

            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(waiter, timeout=5)

            # The crux of the regression: under the leak, the waiter's
            # abandoned executor thread went on to acquire the lock with
            # nobody left to release it, so this third call would block
            # forever. Bound it so a regression fails fast instead of hanging
            # the test itself. The bogus action reaches the trailing
            # public-skill check inside the lock and raises the
            # unsupported-action error, proving the third call really entered
            # the critical section.
            with pytest.raises(ValueError, match="Unsupported action 'bogus'"):
                await asyncio.wait_for(
                    skill_manage_module.skill_manage_tool.coroutine(runtime, "bogus", "cancel-skill"),
                    timeout=5,
                )
        finally:
            # Test-only safety net, independent of the assertions above: under
            # the leak, the lock is left permanently locked with a background
            # thread (from whichever caller's orphaned acquisition landed
            # last) still parked on a *subsequent* acquire() that will now
            # never return. asyncio.run()'s own teardown joins every thread
            # the default executor ever created before it returns, so leaving
            # that thread stuck would hang this test process at
            # interpreter/loop shutdown even after the failure above is
            # already reported. Forcing the lock open here lets any such
            # thread finish so the process can exit; it is a no-op once the
            # fix keeps the lock correctly balanced.
            if lock.locked():
                lock.release()

    asyncio.run(scenario())

    # The cancelled waiter must never reach the critical section's storage
    # lookup; only the third call did.
    assert entered == ["cancel-skill"]


def test_scan_or_raise_does_not_attach_model_tracing(monkeypatch, tmp_path):
    """``_scan_or_raise`` is the in-graph choke point for the skill security scan.

    The graph root already attached the tracing callbacks, so the scan model must
    not attach them again: double-attaching emits duplicate spans and blocks the
    Langfuse handler's ``propagate_attributes`` path, so session_id/user_id never
    reach the trace. Drives the real ``scan_skill_content`` rather than stubbing it,
    so the flag is pinned all the way to the model factory.
    """
    config = _make_config(tmp_path / "skills")
    monkeypatch.setattr("deerflow.skills.security_scanner.get_app_config", lambda: config)

    create_kwargs = {}

    class FakeModel:
        async def ainvoke(self, *args, **kwargs):
            return SimpleNamespace(content='{"decision":"allow","reason":"ok"}')

    def _fake_create_chat_model(**kwargs):
        create_kwargs.update(kwargs)
        return FakeModel()

    monkeypatch.setattr("deerflow.skills.security_scanner.create_chat_model", _fake_create_chat_model)

    result = anyio.run(
        lambda: skill_manage_module._scan_or_raise(
            _skill_content("demo-skill"),
            executable=False,
            location="demo-skill/SKILL.md",
        )
    )

    assert result["decision"] == "allow"
    assert create_kwargs["attach_tracing"] is False
