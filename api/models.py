from django.db import models

class ExapmleOmr(models.Model):
    image=models.ImageField(upload_to='example/')
    created_at=models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table='example_omr'


class AnswerOmr(models.Model):
    exam_id = models.IntegerField() 
    image=models.ImageField(upload_to='answer/')
    json_data = models.JSONField(null=True, blank=True)
    created_at=models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table='answer_omr'





class StudentOmr(models.Model):
    exam_id = models.IntegerField()
    image = models.ImageField(upload_to='student_omr/', blank=True, null=True)
    answer = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'student_omr'
        ordering = ['-created_at']

    def __str__(self):
        return f"Exam {self.exam_id} | OMR {self.id}"
