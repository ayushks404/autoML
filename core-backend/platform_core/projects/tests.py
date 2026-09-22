from unittest.mock import patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from rest_framework.test import APIClient

from .models import AIProject, Experiment


class ProjectFlowTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.csv = SimpleUploadedFile(
            "data.csv", b"a,b,target\n1,2,0\n3,4,1\n5,6,0\n7,8,1\n9,10,0\n11,12,1\n13,14,0\n15,16,1\n17,18,0\n19,20,1\n",
            content_type="text/csv",
        )

    @patch("projects.views.requests.post")
    def test_create_project_triggers_engine_and_sets_training(self, mock_post):
        mock_post.return_value.raise_for_status.return_value = None
        resp = self.client.post(
            "/api/projects/",
            {"name": "p1", "problem_description": "test", "dataset_file": self.csv},
            format="multipart",
        )
        self.assertEqual(resp.status_code, 201)
        project = AIProject.objects.get()
        self.assertEqual(project.status, "training")
        self.assertTrue(mock_post.called)

    @patch("projects.views.requests.post", side_effect=Exception("connection refused"))
    def test_create_project_marks_failed_when_engine_unreachable(self, mock_post):
        resp = self.client.post(
            "/api/projects/",
            {"name": "p2", "problem_description": "test", "dataset_file": self.csv},
            format="multipart",
        )
        self.assertEqual(resp.status_code, 201)  # project record itself is still created
        project = AIProject.objects.get()
        self.assertEqual(project.status, "failed")
        self.assertIn("AI engine", project.error_message)

    def test_rejects_non_csv(self):
        bad = SimpleUploadedFile("data.txt", b"not a csv", content_type="text/plain")
        resp = self.client.post(
            "/api/projects/", {"name": "p3", "dataset_file": bad}, format="multipart"
        )
        self.assertEqual(resp.status_code, 400)

    def test_training_result_callback_sets_completed_and_string_insights(self):
        project = AIProject.objects.create(name="p4", dataset_file=self.csv, status="training")
        resp = self.client.post(
            f"/api/projects/training-result/{project.id}/",
            {
                "task_type": "classification",
                "target_column": "target",
                "best_model": "gradient_boosting",
                "metric_name": "roc_auc",
                "metric_value": 0.91,
                "accuracy": 0.85,
                "all_metrics": {"accuracy": 0.85, "roc_auc": 0.91},
                "model_path": "models/model_1.pkl",
                "top_features": [["a", 0.7], ["b", 0.3]],
                "insights": ["LLM insight generation failed"],  # list, as the old buggy engine sent
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        project.refresh_from_db()
        self.assertEqual(project.status, "completed")
        self.assertIsInstance(project.insights, str)  # must be coerced to string, not stored as a list
        self.assertEqual(project.error_message, None)

    def test_training_failed_callback_sets_failed_status(self):
        project = AIProject.objects.create(name="p5", dataset_file=self.csv, status="training")
        resp = self.client.post(
            f"/api/projects/training-failed/{project.id}/",
            {"error": "ValueError: bad target column"},
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        project.refresh_from_db()
        self.assertEqual(project.status, "failed")
        self.assertIn("bad target column", project.error_message)

    def test_experiment_result_logs_experiment(self):
        project = AIProject.objects.create(name="p6", dataset_file=self.csv, status="training")
        resp = self.client.post(
            f"/api/projects/experiment-result/{project.id}/",
            {"model_name": "random_forest", "metric_name": "f1_weighted", "metric_value": 0.77},
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(Experiment.objects.filter(project=project).count(), 1)

    def test_list_projects(self):
        AIProject.objects.create(name="p7", dataset_file=self.csv, status="completed")
        resp = self.client.get("/api/projects/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.json()), 1)
