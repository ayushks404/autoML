from rest_framework import serializers

from .models import AIProject, Experiment

MAX_UPLOAD_MB = 50


class ExperimentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Experiment
        fields = ["id", "model_name", "metric_name", "metric_value", "created_at"]


class AIProjectSerializer(serializers.ModelSerializer):
    experiments = ExperimentSerializer(many=True, read_only=True)

    class Meta:
        model = AIProject
        fields = "__all__"
        read_only_fields = [
            "status", "task_type", "target_column", "best_model", "metric_name",
            "metric_value", "accuracy", "all_metrics", "model_path", "top_features",
            "insights", "error_message", "created_at", "updated_at",
        ]

    def validate_dataset_file(self, value):
        if not value.name.lower().endswith(".csv"):
            raise serializers.ValidationError("Only .csv files are supported right now.")
        if value.size > MAX_UPLOAD_MB * 1024 * 1024:
            raise serializers.ValidationError(f"File too large (max {MAX_UPLOAD_MB}MB).")
        return value

    def validate_name(self, value):
        if not value.strip():
            raise serializers.ValidationError("Project name cannot be empty.")
        return value
