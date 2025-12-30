from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view
from .models import StudentOmr, AnswerOmr
from .serializer import  AnserOmrSerializer,StudentOmrSerializer
from .serializer import AnswerOmrListSerializer
import os
from django.conf import settings
from django.shortcuts import get_object_or_404
from .omr_runner import run_omr
from.omr_engine import process_omr_image
import cv2
import numpy as np



# start ans omr section

@api_view(['POST'])
def upload_answer_omr(request):
    serializer = AnserOmrSerializer(data=request.data)

    if serializer.is_valid():
        obj = serializer.save()

        return Response({
            "id": obj.id,
            "exam_id": obj.exam_id,
            "json_data": obj.json_data,
            "created_at": obj.created_at
        }, status=status.HTTP_201_CREATED)

    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

#image processing
@api_view(['POST'])
def answer_omr(request):
    img_file = request.FILES.get('image')
    if not img_file:
        return Response({"error": "Image missing"}, status=400)

    img_bytes = img_file.read()
    np_img = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        return Response({"error": "Invalid image"}, status=400)

    result = process_omr_image(img)
    return Response(result)


@api_view(['GET'])
def answer_omr_by_exam(request):
    exam_id = request.GET.get('exam_id')

    if not exam_id:
        return Response({'error': 'exam_id required'}, status=400)

    try:
        exam_id = int(exam_id)
    except ValueError:
        return Response({'error': 'exam_id must be integer'}, status=400)

    qs = AnswerOmr.objects.filter(exam_id=exam_id).order_by('created_at')

    serializer = AnswerOmrListSerializer(
        qs,
        many=True,
        context={'request': request}
    )

    return Response({
        'exam_id': exam_id,
        'total': qs.count(),
        'results': serializer.data
    })


@api_view(['PATCH', 'DELETE'])
def answer_omr_detail(request, pk):
    obj = get_object_or_404(AnswerOmr, pk=pk)

    if request.method == 'DELETE':
        obj.delete()
        return Response(
            {"message": "Deleted"},
            status=status.HTTP_204_NO_CONTENT
        )

    serializer = AnserOmrSerializer(
        obj,
        data=request.data,
        partial=True
    )

    if serializer.is_valid():
        updated = serializer.save()
        return Response(
            {
                "id": updated.id,
                "exam_id": updated.exam_id,
                "json_data": updated.json_data,
            },
            status=status.HTTP_200_OK
        )

    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)



#end ans omr




# views.py
@api_view(['POST'])
def upload_student_omr(request):
    serializer = StudentOmrSerializer(
        data=request.data,
        context={'request': request}
    )

    if serializer.is_valid():
        obj = serializer.save()
        return Response({
            "id": obj.id,
            "exam_id": obj.exam_id,
            "image": request.build_absolute_uri(obj.image.url) if obj.image else None,
            "answer": obj.answer,
            "created_at": obj.created_at,
        }, status=201)

    return Response(serializer.errors, status=400)


@api_view(['GET'])
def student_omr_by_exam(request):
    exam_id = request.GET.get('exam_id')

    if not exam_id:
        return Response(
            {"error": "exam_id query parameter is required"},
            status=400
        )

    try:
        exam_id = int(exam_id)
    except ValueError:
        return Response(
            {"error": "exam_id must be integer"},
            status=400
        )

    qs = StudentOmr.objects.filter(exam_id=exam_id)

    data = StudentOmrSerializer(qs, many=True, context={'request': request}).data

    return Response({
        "exam_id": exam_id,
        "total": qs.count(),
        "results": data
    })


from django.shortcuts import get_object_or_404

@api_view(['PATCH', 'DELETE'])
def student_omr_detail(request, pk):
    obj = get_object_or_404(StudentOmr, pk=pk)

    if request.method == 'DELETE':
        obj.delete()
        return Response(status=204)

    serializer = StudentOmrSerializer(
        obj,
        data=request.data,
        partial=True,
        context={'request': request}
    )

    if serializer.is_valid():
        updated = serializer.save()
        return Response({
            "id": updated.id,
            "exam_id": updated.exam_id,
            "answer": updated.answer,
        })

    return Response(serializer.errors, status=400)


#end
#result section

@api_view(['GET'])
def student_result_list_by_exam(request):
    exam_id = request.GET.get('exam_id')

    if not exam_id:
        return Response(
            {"error": "exam_id query parameter is required"},
            status=status.HTTP_400_BAD_REQUEST
        )

    try:
        exam_id = int(exam_id)
    except ValueError:
        return Response(
            {"error": "exam_id must be integer"},
            status=status.HTTP_400_BAD_REQUEST
        )

    # ================= ANSWER OMR (SET WISE) =================
    answer_qs = AnswerOmr.objects.filter(exam_id=exam_id)

    if not answer_qs.exists():
        return Response(
            {"error": "No answer OMR found for this exam"},
            status=status.HTTP_404_NOT_FOUND
        )

    """
    answer_map = {
      "A": {
        "negative_default": 0.25,
        "msq": {
          "1": { "answer": "B", "negative": 0.25 }
        }
      }
    }
    """
    answer_map = {}

    for ans in answer_qs:
        data = ans.json_data or {}
        set_code = data.get("set_code")
        msq = data.get("msq", {})
        neg_default = data.get("negative_default", 0.0)

        if set_code:
            answer_map[set_code] = {
                "negative_default": neg_default,
                "msq": msq
            }

    # ================= STUDENT OMR =================
    students = StudentOmr.objects.filter(exam_id=exam_id)

    grouped_result = {}

    for stu in students:
        stu_data = stu.answer or {}

        roll = stu_data.get("roll")
        set_code = stu_data.get("set_code")
        stu_msq = stu_data.get("msq", {})

        # ❌ invalid / missing set
        if not set_code or set_code not in answer_map:
            continue

        answer_info = answer_map[set_code]
        correct_msq = answer_info["msq"]
        neg_default = answer_info["negative_default"]

        correct = wrong = skipped = 0
        marks = 0.0
        details = {}

        for q, qinfo in correct_msq.items():
            correct_ans = qinfo.get("answer")
            neg_mark = qinfo.get("negative", neg_default)

            student_ans = stu_msq.get(q)

            if student_ans is None or student_ans == "-":
                status_q = "skipped"
                skipped += 1
            elif student_ans == correct_ans:
                status_q = "correct"
                correct += 1
                marks += 1
            else:
                status_q = "wrong"
                wrong += 1
                marks -= float(neg_mark)

            details[q] = {
                "correct": correct_ans,
                "student": student_ans,
                "negative": neg_mark,
                "status": status_q
            }

        result = {
            "student_id": stu.id,
            "roll": roll,
            "set_code": set_code,
            "total_questions": len(correct_msq),
            "correct": correct,
            "wrong": wrong,
            "skipped": skipped,
            "marks": round(marks, 2),
            "details": details,
            "submitted_at": stu.created_at
        }

        grouped_result.setdefault(set_code, []).append(result)

    return Response({
        "exam_id": exam_id,
        "total_students": students.count(),
        "sets": grouped_result
    })


