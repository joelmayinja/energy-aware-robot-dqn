import time
import unittest
from unittest.mock import patch

from src import dashboard


class DashboardServerTests(unittest.TestCase):
    def setUp(self):
        dashboard.controller.reset()
        self.pybullet_patch = patch("src.dashboard.pybullet_available", return_value=True)
        self.pybullet_patch.start()
        self.client = dashboard.app.test_client()

    def tearDown(self):
        dashboard.controller.stop()
        if dashboard.controller.thread is not None:
            dashboard.controller.thread.join(timeout=2)
        self.pybullet_patch.stop()

    def test_snapshot_is_a_detached_copy(self):
        dashboard.controller._update_step(
            {
                "episode": 1,
                "episode_step": 1,
                "baseline_grid_draw": 0.5,
                "agent_grid_draw": 0.25,
            }
        )
        snapshot = dashboard.controller.snapshot()
        snapshot["series"][0]["agent"] = 99

        self.assertEqual(dashboard.controller.snapshot()["series"][0]["agent"], 0.25)

    def test_starting_new_run_gets_new_run_id_and_clears_previous_history(self):
        dashboard.controller._record_episode({"episode": 1})

        with patch("src.dashboard.train", return_value=None):
            self.assertTrue(dashboard.controller.start(1))
            dashboard.controller.thread.join(timeout=2)
        snapshot = dashboard.controller.snapshot()

        self.assertEqual(snapshot["status"], "completed")
        self.assertEqual(snapshot["run_id"], 1)
        self.assertEqual(snapshot["history"], [])

    def test_dashboard_and_status_are_served(self):
        page = self.client.get("/")
        try:
            self.assertEqual(page.status_code, 200)
            self.assertIn(b"Start teacher training", page.data)
            self.assertIn(b"fixed-speed baseline", page.data)
            self.assertIn(b"Upload and validate", page.data)
        finally:
            page.close()

        response = self.client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["status"], "idle")

    def test_invalid_training_request_returns_actionable_error(self):
        response = self.client.post("/api/train", json={"episodes": 501})

        self.assertEqual(response.status_code, 400)
        self.assertIn("1 and 500 episodes", response.json["error"])

    def test_training_is_blocked_when_pybullet_is_unavailable(self):
        with patch("src.dashboard.pybullet_available", return_value=False):
            status = self.client.get("/api/status")
            response = self.client.post("/api/train", json={"episodes": 1})

        self.assertFalse(status.json["pybullet_available"])
        self.assertEqual(response.status_code, 503)
        self.assertIn("PyBullet is unavailable", response.json["error"])

    def test_speed_endpoint_validates_range(self):
        response = self.client.post("/api/speed", json={"speed": 0})

        self.assertEqual(response.status_code, 400)
        self.assertIn("from 1 to 5", response.json["error"])

    def test_training_endpoint_reports_live_metrics_and_completion(self):
        def fake_train(
            episodes,
            stop_event,
            pause_event,
            speed_provider,
            on_step,
            on_episode,
            resume_model=False,
        ):
            on_step(
                {
                    "stage": "baseline",
                    "episode": 1,
                    "episode_step": 1,
                    "baseline_grid_draw": 0.5,
                    "agent_grid_draw": None,
                }
            )
            on_step(
                {
                    "stage": "teacher",
                    "episode": 1,
                    "episode_step": 1,
                    "baseline_grid_draw": 0.5,
                    "agent_grid_draw": 0.25,
                }
            )
            on_episode({"episode": 1, "reward": 2.0, "grid_energy": 0.5})

        with patch("src.dashboard.train", side_effect=fake_train):
            response = self.client.post("/api/train", json={"episodes": 1})
            self.assertEqual(response.status_code, 202)

            deadline = time.monotonic() + 2
            while time.monotonic() < deadline and dashboard.controller.snapshot()["status"] == "running":
                time.sleep(0.01)
            dashboard.controller.thread.join(timeout=2)

        metrics = self.client.get("/api/status").json
        self.assertEqual(metrics["status"], "completed")
        self.assertEqual(metrics["series"], [{"step": 1, "agent": 0.25, "baseline": 0.5}])
        self.assertEqual(metrics["history"][0]["grid_energy"], 0.5)

    def test_model_upload_validates_and_saves_teacher_weights(self):
        import io
        import tempfile
        from pathlib import Path

        import torch

        from src.dqn_teacher import DQNAgent

        agent = DQNAgent(state_dim=23, num_joints=6, device="cpu")
        model_file = io.BytesIO()
        torch.save(agent.q_network.state_dict(), model_file)
        model_file.seek(0)

        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "teacher.pt"
            with patch("src.dashboard.MODEL_PATH", model_path):
                response = self.client.post(
                    "/api/model/upload",
                    data={"model": (model_file, "teacher.pt")},
                    content_type="multipart/form-data",
                )

                self.assertEqual(response.status_code, 200, response.json)
                self.assertTrue(model_path.is_file())
                self.assertTrue(response.json["model_available"])

    def test_model_upload_rejects_unrecognized_extension(self):
        import io

        response = self.client.post(
            "/api/model/upload",
            data={"model": (io.BytesIO(b"weights"), "teacher.bin")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn(".pt or .pth", response.json["error"])

    def test_model_upload_rejects_incompatible_weights(self):
        import io
        import tempfile
        from pathlib import Path

        import torch

        model_file = io.BytesIO()
        torch.save({"unexpected": torch.zeros(1)}, model_file)
        model_file.seek(0)

        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "teacher.pt"
            with patch("src.dashboard.MODEL_PATH", model_path):
                response = self.client.post(
                    "/api/model/upload",
                    data={"model": (model_file, "wrong-shape.pt")},
                    content_type="multipart/form-data",
                )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid teacher weights", response.json["error"])
        self.assertFalse(model_path.exists())

    def test_training_rejects_resume_without_saved_model(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary_directory:
            with patch("src.dashboard.MODEL_PATH", Path(temporary_directory) / "missing.pt"):
                response = self.client.post(
                    "/api/train", json={"episodes": 1, "resume_model": True}
                )

        self.assertEqual(response.status_code, 409)
        self.assertIn("No saved teacher model", response.json["error"])

    def test_training_can_continue_from_a_saved_teacher(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        received_resume_values = []

        def fake_train(
            episodes,
            stop_event,
            pause_event,
            speed_provider,
            on_step,
            on_episode,
            resume_model=False,
        ):
            received_resume_values.append(resume_model)

        with TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "teacher.pt"
            model_path.touch()
            with (
                patch("src.dashboard.MODEL_PATH", model_path),
                patch("src.dashboard.train", side_effect=fake_train),
            ):
                response = self.client.post(
                    "/api/train", json={"episodes": 1, "resume_model": True}
                )
                dashboard.controller.thread.join(timeout=2)

        self.assertEqual(response.status_code, 202)
        self.assertTrue(response.json["resumed_model"])
        self.assertEqual(received_resume_values, [True])

    def test_pause_resume_and_stop_are_available_during_training(self):
        def waiting_train(
            episodes,
            stop_event,
            pause_event,
            speed_provider,
            on_step,
            on_episode,
            resume_model=False,
        ):
            stop_event.wait(timeout=2)

        with patch("src.dashboard.train", side_effect=waiting_train):
            self.assertEqual(self.client.post("/api/train", json={"episodes": 1}).status_code, 202)
            self.assertEqual(self.client.post("/api/pause").json["status"], "paused")
            self.assertEqual(self.client.post("/api/resume").json["status"], "running")
            self.assertEqual(self.client.post("/api/stop").json["status"], "stopping")


if __name__ == "__main__":
    unittest.main()
