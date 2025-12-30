from rest_framework import serializers
from .models import ExapmleOmr,AnswerOmr, StudentOmr
from django.core.files.storage import default_storage
import os
import cv2
import numpy as np



    
#ati section holo ans omr upload
from rest_framework import serializers
from .models import AnswerOmr


class AnserOmrSerializer(serializers.ModelSerializer):

    class Meta:
        model = AnswerOmr
        fields = ['id', 'exam_id', 'json_data', 'created_at']
        read_only_fields = ['id', 'created_at']

    def validate(self, attrs):
        # exam_id validation (optional)
        exam_id = attrs.get('exam_id')
        if exam_id is not None:
            try:
                attrs['exam_id'] = int(exam_id)
            except ValueError:
                raise serializers.ValidationError({
                    'exam_id': 'Must be integer'
                })

        return attrs

    def update(self, instance, validated_data):
        """
        PARTIAL UPDATE LOGIC
        """

        # -------- exam_id --------
        if 'exam_id' in validated_data:
            instance.exam_id = validated_data['exam_id']

        # -------- json_data --------
        if 'json_data' in validated_data:
            incoming = validated_data.get('json_data') or {}
            existing = instance.json_data or {}

            # set_code
            if 'set_code' in incoming:
                existing['set_code'] = incoming['set_code']

            # negative_default
            if 'negative_default' in incoming:
                existing['negative_default'] = incoming['negative_default']

            # msq (deep merge)
            if 'msq' in incoming:
                existing_msq = existing.get('msq', {})
                incoming_msq = incoming.get('msq', {})

                for q_no, q_data in incoming_msq.items():
                    if not isinstance(q_data, dict):
                        continue

                    prev = existing_msq.get(str(q_no), {})
                    existing_msq[str(q_no)] = {
                        "answer": q_data.get(
                            'answer', prev.get('answer')
                        ),
                        "negative": q_data.get(
                            'negative', prev.get('negative')
                        )
                    }

                existing['msq'] = existing_msq

            instance.json_data = existing

        instance.save()
        return instance


    

class AnswerOmrListSerializer(serializers.ModelSerializer):
    image = serializers.SerializerMethodField()

    class Meta:
        model = AnswerOmr
        fields = ['id', 'exam_id', 'image', 'json_data', 'created_at']

    def get_image(self, obj):
        request = self.context.get('request')
        if obj.image:
            if request:
                return request.build_absolute_uri(obj.image.url)
            return obj.image.url
        return None

#end 

#start student

class StudentOmrSerializer(serializers.ModelSerializer):

    class Meta:
        model = StudentOmr
        fields = ['id', 'exam_id', 'image', 'answer', 'created_at']
        read_only_fields = ['id', 'created_at']

    def validate(self, attrs):
        image = attrs.get('image')
        exam_id = attrs.get('exam_id')

        if exam_id is None:
            raise serializers.ValidationError({
                'exam_id': 'exam_id is required'
            })

        try:
            attrs['exam_id'] = int(exam_id)
        except ValueError:
            raise serializers.ValidationError({
                'exam_id': 'exam_id must be integer'
            })

        if image:
            file_bytes = np.frombuffer(image.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                raise serializers.ValidationError({
                    'image': 'Invalid image file'
                })
            image.seek(0)

        return attrs

    # 🔥 MERGE ANSWER JSON
    def update(self, instance, validated_data):

        if 'exam_id' in validated_data:
            instance.exam_id = validated_data['exam_id']

        if 'image' in validated_data:
            instance.image = validated_data['image']

        if 'answer' in validated_data:
            incoming = validated_data['answer'] or {}
            existing = instance.answer or {}

            # merge everything
            for key, value in incoming.items():
                existing[key] = value

            instance.answer = existing

        instance.save()
        return instance