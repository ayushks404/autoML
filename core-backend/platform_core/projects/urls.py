from django.urls import path

from .views import (
    ProjectDetailAPI,
    ProjectListCreateAPI,
    experiment_result,
    training_failed,
    training_result,
)

urlpatterns = [
    path("", ProjectListCreateAPI.as_view()),
    path("<int:project_id>/", ProjectDetailAPI.as_view()),
    path("training-result/<int:project_id>/", training_result),
    path("training-failed/<int:project_id>/", training_failed),
    path("experiment-result/<int:project_id>/", experiment_result),
]
