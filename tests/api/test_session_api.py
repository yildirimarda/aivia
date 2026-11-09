"""Session HTTP routes: create, message, preview, history and asset access.

Every route is scoped to one session: a session's assets, plans and current
composition are only visible through the session that owns them.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from aivia.api import create_app
from aivia.core.config import ApiConfig

KEY = "test-api-key"
AUTH = {"X-API-Key": KEY}


def make_app(**overrides: object) -> TestClient:
    values: dict[str, object] = {
        "require_auth": True,
        "api_key_ref": "AIVA_API_KEY",
    }
    values.update(overrides)
    config = ApiConfig.model_validate(values)
    app = create_app(config, environ={"AIVA_API_KEY": KEY})
    return TestClient(app)


def _session_body(**fields: object) -> dict[str, object]:
    body: dict[str, object] = {}
    body.update(fields)
    return body


def test_create_session_returns_the_session() -> None:
    client = make_app()

    response = client.post(
        "/api/v1/sessions",
        json=_session_body(title="intro", run_id="run_1", profile="default"),
        headers=AUTH,
    )

    assert response.status_code == 201
    body = response.json()
    assert body["id"].startswith("session_")
    assert body["session_id"] == body["id"]
    assert body["title"] == "intro"
    assert body["run_id"] == "run_1"
    assert body["profile"] == "default"
    assert body["assets"] == []
    assert body["history"] == []
    assert body["plan_count"] == 0


def test_create_session_accepts_assets() -> None:
    client = make_app()

    response = client.post(
        "/api/v1/sessions",
        json=_session_body(
            title="intro",
            assets=[
                {"label": "hero", "kind": "image", "path": "/registry/hero.png"},
                {"label": "logo", "kind": "audio", "path": "/registry/logo.mp3"},
            ],
        ),
        headers=AUTH,
    )

    assert response.status_code == 201
    body = response.json()
    assert [asset["label"] for asset in body["assets"]] == ["hero", "logo"]
    assert body["asset_count"] == 2


def test_create_session_uses_aliases() -> None:
    client = make_app()

    response = client.post(
        "/api/v1/sessions",
        json=_session_body(
            session_id="session_alias",
            name="titled",
            run="run_x",
            profile_name="prof",
        ),
        headers=AUTH,
    )

    assert response.status_code == 201
    body = response.json()
    assert body["id"] == "session_alias"
    assert body["title"] == "titled"
    assert body["run_id"] == "run_x"
    assert body["profile"] == "prof"


def test_message_appends_a_plan_to_history() -> None:
    client = make_app()
    created = client.post(
        "/api/v1/sessions",
        json=_session_body(title="intro"),
        headers=AUTH,
    ).json()
    session_id = created["id"]

    response = client.post(
        f"/api/v1/sessions/{session_id}/message",
        json={"message": "first draft"},
        headers=AUTH,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["plan_count"] == 1
    assert body["history"][0]["instruction"] == "first draft"
    assert body["history"][0]["id"].startswith("plan_")


def test_message_accepts_text_and_instruction_aliases() -> None:
    client = make_app()
    session_id = client.post(
        "/api/v1/sessions",
        json=_session_body(title="intro"),
        headers=AUTH,
    ).json()["id"]

    response = client.post(
        f"/api/v1/sessions/{session_id}/message",
        json={"instruction": "revised cut"},
        headers=AUTH,
    )

    assert response.status_code == 200
    assert response.json()["history"][0]["instruction"] == "revised cut"


def test_message_requires_a_non_empty_message() -> None:
    client = make_app()
    session_id = client.post(
        "/api/v1/sessions",
        json=_session_body(title="intro"),
        headers=AUTH,
    ).json()["id"]

    response = client.post(
        f"/api/v1/sessions/{session_id}/message",
        json={"message": "   "},
        headers=AUTH,
    )

    assert response.status_code == 422


def test_preview_returns_the_current_composition() -> None:
    client = make_app()
    session_id = client.post(
        "/api/v1/sessions",
        json=_session_body(title="intro"),
        headers=AUTH,
    ).json()["id"]

    response = client.get(
        f"/api/v1/sessions/{session_id}/preview",
        headers=AUTH,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == session_id
    assert "content_hash" in body["current"]


def test_history_returns_the_ordered_plans() -> None:
    client = make_app()
    session_id = client.post(
        "/api/v1/sessions",
        json=_session_body(title="intro"),
        headers=AUTH,
    ).json()["id"]
    client.post(
        f"/api/v1/sessions/{session_id}/message",
        json={"message": "first"},
        headers=AUTH,
    )
    client.post(
        f"/api/v1/sessions/{session_id}/message",
        json={"message": "second"},
        headers=AUTH,
    )

    response = client.get(
        f"/api/v1/sessions/{session_id}/history",
        headers=AUTH,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == session_id
    assert body["plan_count"] == 2
    assert [plan["instruction"] for plan in body["history"]] == ["first", "second"]


def test_assets_are_scoped_to_the_session() -> None:
    client = make_app()
    first = client.post(
        "/api/v1/sessions",
        json=_session_body(
            title="one",
            assets=[{"label": "hero", "kind": "image", "path": "/registry/hero.png"}],
        ),
        headers=AUTH,
    ).json()["id"]
    second = client.post(
        "/api/v1/sessions",
        json=_session_body(title="two"),
        headers=AUTH,
    ).json()["id"]

    first_assets = client.get(f"/api/v1/sessions/{first}/assets", headers=AUTH).json()
    second_assets = client.get(f"/api/v1/sessions/{second}/assets", headers=AUTH).json()

    assert first_assets["session_id"] == first
    assert [asset["label"] for asset in first_assets["assets"]] == ["hero"]
    assert first_assets["asset_count"] == 1
    assert second_assets["session_id"] == second
    assert second_assets["assets"] == []
    assert second_assets["asset_count"] == 0


def test_get_asset_by_label_is_scoped_to_the_session() -> None:
    client = make_app()
    first = client.post(
        "/api/v1/sessions",
        json=_session_body(
            title="one",
            assets=[{"label": "hero", "kind": "image", "path": "/registry/hero.png"}],
        ),
        headers=AUTH,
    ).json()["id"]
    second = client.post(
        "/api/v1/sessions",
        json=_session_body(title="two"),
        headers=AUTH,
    ).json()["id"]

    found = client.get(f"/api/v1/sessions/{first}/assets/hero", headers=AUTH)
    assert found.status_code == 200
    assert found.json()["session_id"] == first
    assert found.json()["asset"]["label"] == "hero"

    missing = client.get(f"/api/v1/sessions/{first}/assets/absent", headers=AUTH)
    assert missing.status_code == 404

    cross_session = client.get(f"/api/v1/sessions/{second}/assets/hero", headers=AUTH)
    assert cross_session.status_code == 404


def test_unknown_session_is_not_found() -> None:
    client = make_app()

    assert (
        client.post(
            "/api/v1/sessions/session_missing/message",
            json={"message": "hello"},
            headers=AUTH,
        ).status_code
        == 404
    )
    assert (
        client.get("/api/v1/sessions/session_missing/preview", headers=AUTH).status_code
        == 404
    )
    assert (
        client.get("/api/v1/sessions/session_missing/history", headers=AUTH).status_code
        == 404
    )
    assert (
        client.get("/api/v1/sessions/session_missing/assets", headers=AUTH).status_code
        == 404
    )
    assert (
        client.get(
            "/api/v1/sessions/session_missing/assets/hero", headers=AUTH
        ).status_code
        == 404
    )


def test_session_routes_require_authentication() -> None:
    client = TestClient(
        create_app(
            ApiConfig(require_auth=True, api_key_ref="AIVA_API_KEY"),
            environ={"AIVA_API_KEY": KEY},
        )
    )

    assert client.post("/api/v1/sessions", json={"title": "intro"}).status_code == 401
    authorized = client.post(
        "/api/v1/sessions",
        json={"title": "intro"},
        headers=AUTH,
    )
    assert authorized.status_code == 201
