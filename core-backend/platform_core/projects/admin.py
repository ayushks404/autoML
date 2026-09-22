from django.contrib import admin

from .models import AIProject, Experiment


@admin.register(AIProject)
class AIProjectAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "status", "task_type", "best_model", "metric_name", "metric_value", "created_at")
    list_filter = ("status", "task_type")
    search_fields = ("name",)


@admin.register(Experiment)
class ExperimentAdmin(admin.ModelAdmin):
    list_display = ("id", "project", "model_name", "metric_name", "metric_value", "created_at")
    list_filter = ("model_name",)
