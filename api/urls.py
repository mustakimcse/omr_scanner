
from django.urls import path
from . import views


urlpatterns = [
    
    path('answer-omr/preview', views.answer_omr),

    path('answer-omr', views.upload_answer_omr),
    path('answer-omr/by-exam', views.answer_omr_by_exam,),
    path('answer-omr/<int:pk>/', views.answer_omr_detail, name='answer-omr-detail'),

    path('student-omr', views.upload_student_omr),
    path('student-omr/by-exam', views.student_omr_by_exam),
    path('student-omr/<int:pk>/', views.student_omr_detail),


    path('student-result/by-exam',views.student_result_list_by_exam),


]
