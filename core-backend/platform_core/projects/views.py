import logging

import requests
from django.conf import settings
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import AIProject, Experiment
from .serializers import AIProjectSerializer

logger = logging.getLogger("projects")


class ProjectListCreateAPI(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def get(self, request):
        projects = AIProject.objects.all()
        return Response(AIProjectSerializer(projects, many=True).data)

    def post(self, request):
        serializer = AIProjectSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        project = serializer.save(status="training")

        try:
            resp = requests.post(
                f"{settings.AI_ENGINE_URL}/start-training/{project.id}",
                json={"dataset_path": project.dataset_file.path},
                timeout=5,  # just enough to confirm the engine accepted the job; training itself runs after
            )
            resp.raise_for_status()
        except Exception as e:  # network errors, timeouts, DNS failures, bad status codes, etc.
            logger.error("AI engine unreachable for project %s: %s", project.id, e)
            project.status = "failed"
            project.error_message = f"Could not reach AI engine at {settings.AI_ENGINE_URL}: {e}"
            project.save(update_fields=["status", "error_message"])
            return Response(
                {
                    "message": "Project created, but the AI engine could not be reached. "
                               "Is it running on the configured AI_ENGINE_URL?",
                    "project": AIProjectSerializer(project).data,
                },
                status=status.HTTP_201_CREATED,
            )

        return Response(
            {"message": "Project created & training started.", "project": AIProjectSerializer(project).data},
            status=status.HTTP_201_CREATED,
        )


class ProjectDetailAPI(APIView):
    def get(self, request, project_id):
        try:
            project = AIProject.objects.get(id=project_id)
        except AIProject.DoesNotExist:
            return Response({"error": "Project not found"}, status=status.HTTP_404_NOT_FOUND)
        return Response(AIProjectSerializer(project).data)


@api_view(["POST"])
def training_result(request, project_id):
    try:
        project = AIProject.objects.get(id=project_id)
    except AIProject.DoesNotExist:
        return Response({"error": "Project not found"}, status=status.HTTP_404_NOT_FOUND)

    data = request.data

    project.status = "completed"
    project.task_type = data.get("task_type")
    project.target_column = data.get("target_column")
    project.best_model = data.get("best_model")
    project.metric_name = data.get("metric_name")
    project.metric_value = data.get("metric_value")
    project.accuracy = data.get("accuracy")
    project.all_metrics = data.get("all_metrics")
    project.model_path = data.get("model_path")
    project.top_features = data.get("top_features")

    insights = data.get("insights")
    if isinstance(insights, list):  # defensive: always store as text regardless of caller
        insights = "\n".join(str(i) for i in insights)
    project.insights = insights

    project.error_message = None
    project.save()

    return Response({"message": "result saved"})


@api_view(["POST"])
def training_failed(request, project_id):
    try:
        project = AIProject.objects.get(id=project_id)
    except AIProject.DoesNotExist:
        return Response({"error": "Project not found"}, status=status.HTTP_404_NOT_FOUND)

    project.status = "failed"
    project.error_message = request.data.get("error", "Unknown error during training.")
    project.save(update_fields=["status", "error_message"])
    logger.warning("Project %s training failed: %s", project_id, project.error_message)
    return Response({"message": "failure recorded"})


@api_view(["POST"])
def experiment_result(request, project_id):
    try:
        project = AIProject.objects.get(id=project_id)
    except AIProject.DoesNotExist:
        return Response({"error": "Project not found"}, status=status.HTTP_404_NOT_FOUND)

    Experiment.objects.create(
        project=project,
        model_name=request.data.get("model_name", "unknown"),
        metric_name=request.data.get("metric_name", "accuracy"),
        metric_value=request.data.get("metric_value"),
    )
    return Response({"message": "experiment logged"})
