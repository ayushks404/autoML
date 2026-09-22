from django.db import models


class AIProject(models.Model):
    STATUS_CHOICES = [
        ("created", "Created"),
        ("training", "Training"),
        ("completed", "Completed"),
        ("failed", "Failed"),
    ]
    TASK_CHOICES = [
        ("classification", "Classification"),
        ("regression", "Regression"),
    ]

    name = models.CharField(max_length=255)
    problem_description = models.TextField(blank=True, default="")
    dataset_file = models.FileField(upload_to="datasets/")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="created")

    # auto-detected by the AI engine, not chosen by the user
    task_type = models.CharField(max_length=20, choices=TASK_CHOICES, null=True, blank=True)
    target_column = models.CharField(max_length=255, null=True, blank=True)

    best_model = models.CharField(max_length=100, null=True, blank=True)
    metric_name = models.CharField(max_length=50, null=True, blank=True)
    metric_value = models.FloatField(null=True, blank=True)
    accuracy = models.FloatField(null=True, blank=True)  # convenience copy, classification only
    all_metrics = models.JSONField(null=True, blank=True)

    model_path = models.CharField(max_length=500, null=True, blank=True)
    top_features = models.JSONField(null=True, blank=True)
    insights = models.TextField(null=True, blank=True)  # always plain text, never a list

    error_message = models.TextField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.name} ({self.status})"


class Experiment(models.Model):
    project = models.ForeignKey(AIProject, on_delete=models.CASCADE, related_name="experiments")
    model_name = models.CharField(max_length=200)
    metric_name = models.CharField(max_length=50, default="accuracy")
    metric_value = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.project.name} / {self.model_name}: {self.metric_name}={self.metric_value}"
