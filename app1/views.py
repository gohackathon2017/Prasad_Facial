import os
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from .models import Student, Attendance, CameraConfiguration
from django.core.files.base import ContentFile
from datetime import datetime, timedelta
from django.utils import timezone
import pygame  # Import pygame for playing sounds
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.urls import reverse_lazy
from django.contrib.auth.decorators import login_required
import threading
import time
import base64
from django.db import IntegrityError
from django.contrib.auth.decorators import login_required, user_passes_test
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth import authenticate, login
from django.contrib import messages
from .models import Student
from .models import CameraConfiguration
from datetime import datetime
import requests
from django.core.files.storage import default_storage
from .services import get_mtcnn, get_resnet


# Initialize MTCNN and InceptionResnetV1
mtcnn = MTCNN(keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()


# Function to detect and encode faces
import dropbox
import cv2
import numpy as np
import torch
from django.conf import settings
from django.core.exceptions import ValidationError
from PIL import Image
from io import BytesIO


# Initialize Dropbox client
def get_dropbox_client():
    ACCESS_TOKEN = 'sl.CAr60NQ0zDHfQIDl3TEcQQ6SyCZbJ09Gcn35Ir1qP-Zl7FuATohWoLZwp6j2m8IlGHJkXBd1Ty-v2ikRRZuxsHwuRK07l2UzJRpJnHamyXnvpsXKeMytuNl9v0e9Jrz0jzqmgskY6J9_GtqR0IgXbgw'
    dbx = dropbox.Dropbox(ACCESS_TOKEN)
    return dbx


# Function to download the image from Dropbox
def download_image_from_dropbox(dropbox_path):
    dbx = get_dropbox_client()
    
    try:
        # Download file from Dropbox
        _, res = dbx.files_download(path=dropbox_path)
        image_data = res.content

        # Convert binary image data to numpy array and then to OpenCV format
        image = np.array(bytearray(image_data), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image
    except dropbox.exceptions.ApiError as e:
        raise ValidationError(f"Error downloading image from Dropbox: {str(e)}")


# Function to detect and encode face from image
def detect_and_encode(image):

    mtcnn = get_mtcnn()  # Load preloaded MTCNN model
    resnet = get_resnet()  # Load preloaded InceptionResnetV1 model
    with torch.no_grad():
        boxes, _ = mtcnn.detect(image)
        if boxes is not None:
            faces = []
            for box in boxes:
                face = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                if face.size == 0:
                    continue
                face = cv2.resize(face, (160, 160))
                face = np.transpose(face, (2, 0, 1)).astype(np.float32) / 255.0
                face_tensor = torch.tensor(face).unsqueeze(0)
                encoding = resnet(face_tensor).detach().numpy().flatten()
                faces.append(encoding)
            return faces
    return []


# Function to encode uploaded images
def encode_uploaded_images():
    known_face_encodings = []
    known_face_names = []

    # Fetch only authorized images
    uploaded_images = Student.objects.filter(authorized=True)

    for student in uploaded_images:
        dropbox_path = f"{student.image.name}"  # Dropbox path (assuming image path is stored as relative path)

        try:
            known_image = download_image_from_dropbox(dropbox_path)
            print(dropbox_path)  # Download the image from Dropbox
            known_image_rgb = cv2.cvtColor(known_image, cv2.COLOR_BGR2RGB)
            encodings = detect_and_encode(known_image_rgb)
            if encodings:
                known_face_encodings.extend(encodings)
                known_face_names.append(student.name)
        except ValidationError as e:
            print(f"Error processing image for student {student.name}: {e}")
            continue

    return known_face_encodings, known_face_names


# Function to recognize faces
def recognize_faces(known_encodings, known_names, test_encodings, threshold=0.6):
    recognized_names = []
    for test_encoding in test_encodings:
        distances = np.linalg.norm(known_encodings - test_encoding, axis=1)
        min_distance_idx = np.argmin(distances)
        if distances[min_distance_idx] < threshold:
            recognized_names.append(known_names[min_distance_idx])
        else:
            recognized_names.append('Not Recognized')
    return recognized_names


# View for capturing student information and image
import base64
from django.core.files.base import ContentFile
from django.shortcuts import render, redirect
from django.core.exceptions import ValidationError
from django.http import JsonResponse
import dropbox
from .models import Student  # Adjust if the model import path differs

# Dropbox API initialization


def capture_student(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        fname = request.POST.get('fname')
        rollno = request.POST.get('rollno')
        email = request.POST.get('email')
        phone_number = request.POST.get('phone_number')
        batch = request.POST.get('batch')
        phase = request.POST.get('phase')
        image_data = request.POST.get('image_data')

        # Decode the base64 image data if provided
        if image_data:
            try:
                # Split the data to separate the header from the encoded part
                header, encoded = image_data.split(',', 1)
                image_file = ContentFile(base64.b64decode(encoded), name=f"{name}.jpg")

                # Define the path in Dropbox
                media_path = f"/media/{image_file.name}"
                print(f"Saving file to Dropbox with path: {media_path}")

                # Initialize Dropbox client
                dbx = get_dropbox_client()

                # Upload the file to Dropbox
                dbx.files_upload(image_file.read(), media_path)
                
                # Store only the Dropbox path in the database
                file_name = media_path  # Dropbox path

                # Create the Student object and save to the database
                student = Student(
                    name=name,
                    fname=fname,
                    rollno=rollno,
                    email=email,
                    phone_number=phone_number,
                    batch=batch,
                    phase=phase,
                    image=file_name,  # Store the Dropbox path in the image field
                    authorized=False
                )
                student.save()

                print("File saved at Dropbox path:", file_name)

                return redirect('selfie_success')

            except Exception as e:
                raise ValidationError("Error in image processing: " + str(e))

    return render(request, 'capture_student.html')





# Success view after capturing student information and image
def selfie_success(request):
    return render(request, 'selfie_success.html')


# This views for capturing studen faces and recognize
import dropbox
from io import BytesIO
import cv2
import numpy as np
import threading
import time
import pygame
from django.utils import timezone
from django.shortcuts import render, redirect
from .models import Student, Attendance


# Initialize Dropbox API client
dbx = dropbox.Dropbox('sl.CAr60NQ0zDHfQIDl3TEcQQ6SyCZbJ09Gcn35Ir1qP-Zl7FuATohWoLZwp6j2m8IlGHJkXBd1Ty-v2ikRRZuxsHwuRK07l2UzJRpJnHamyXnvpsXKeMytuNl9v0e9Jrz0jzqmgskY6J9_GtqR0IgXbgw')  # Replace with your Dropbox access token

# Function to download an image from Dropbox
def download_image_from_dropbox(dropbox_path):
    try:
        metadata, res = dbx.files_download(path=dropbox_path)
        image_data = res.content
        image = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image
    except dropbox.exceptions.ApiError as e:
        print(f"Error downloading file from Dropbox: {e}")
        return None

def capture_and_recognize(request):
    stop_events = []  # List to store stop events for each thread
    camera_threads = []  # List to store threads for each camera
    camera_windows = []  # List to store window names
    error_messages = []  # List to capture errors from threads

    # Capture subject from frontend form submission
    subject_name = request.POST.get('subject')  # Assuming subject is passed in a POST request

    def process_frame(cam_config, stop_event):
        """Thread function to capture and process frames for each camera."""
        cap = None
        window_created = False  # Flag to track if the window was created
        try:
            # Check if the camera source is a number (local webcam) or a string (IP camera URL)
            if cam_config.camera_source.isdigit():
                cap = cv2.VideoCapture(int(cam_config.camera_source))  # Use integer index for webcam
            else:
                cap = cv2.VideoCapture(cam_config.camera_source)  # Use string for IP camera URL

            if not cap.isOpened():
                raise Exception(f"Unable to access camera {cam_config.name}.")

            threshold = cam_config.threshold

            # Initialize pygame mixer for sound playback
            pygame.mixer.init()
            success_sound = pygame.mixer.Sound('app1/suc.wav')  # load sound path

            window_name = f'Face Recognition - {cam_config.name}'
            camera_windows.append(window_name)  # Track the window name

            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    print(f"Failed to capture frame for camera: {cam_config.name}")
                    break  # If frame capture fails, break from the loop

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                test_face_encodings = detect_and_encode(frame_rgb)  # Function to detect and encode face in frame

                if test_face_encodings:
                    known_face_encodings, known_face_names = encode_uploaded_images()  # Load known face encodings once
                    if known_face_encodings:
                        names = recognize_faces(np.array(known_face_encodings), known_face_names, test_face_encodings, threshold)

                        for name, box in zip(names, mtcnn.detect(frame_rgb)[0]):
                            if box is not None:
                                (x1, y1, x2, y2) = map(int, box)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                                if name != 'Not Recognized':
                                    students = Student.objects.filter(name=name)
                                    if students.exists():
                                        student = students.first()

                                        # Retrieve today's attendance record or create a new one if it doesn't exist
                                        today = timezone.now().date()
                                        try:
                                            attendance = Attendance.objects.get(roll_number=student, date=today)
                                        except Attendance.DoesNotExist:
                                            attendance = None

                                        if attendance:
                                            # Display student name and mark as "Present" every time they appear
                                            attendance.student_name = student.name
                                            attendance.Subject = subject_name if subject_name else 'Unknown'
                                            attendance.mark_checked_in()  # Consider updating this to reflect repeated presence instead of a one-time check-in
                                            success_sound.play()
                                            cv2.putText(frame, f"{name} - Present", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                                        else:
                                            # If no attendance record exists, create a new one and mark as "Present"
                                            attendance = Attendance(
                                                roll_number=student,
                                                student_name=student.name,
                                                Subject=subject_name if subject_name else 'Unknown',
                                                date=today
                                            )
                                            attendance.mark_checked_in()
                                            success_sound.play()
                                            cv2.putText(frame, f"{name} - Present", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                                        
                                        # Save the attendance object
                                        attendance.save()

                                        # Send attendance data to manual project API
                                        send_attendance_to_manual_project(student.rollno, subject_name, student.phase)

                # Display frame in separate window for each camera
                if not window_created:
                    cv2.namedWindow(window_name)  # Only create window once
                    window_created = True  # Mark window as created
                
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_event.set()  # Signal the thread to stop when 'q' is pressed
                    break

        except Exception as e:
            print(f"Error in thread for {cam_config.name}: {e}")
            error_messages.append(str(e))  # Capture error message
        finally:
            if cap is not None:
                cap.release()
            if window_created:
                cv2.destroyWindow(window_name)  # Only destroy if window was created

    try:
        # Get all camera configurations
        cam_configs = CameraConfiguration.objects.all()
        if not cam_configs.exists():
            raise Exception("No camera configurations found. Please configure them in the admin panel.")

        # Create threads for each camera configuration
        for cam_config in cam_configs:
            stop_event = threading.Event()
            stop_events.append(stop_event)

            camera_thread = threading.Thread(target=process_frame, args=(cam_config, stop_event))
            camera_threads.append(camera_thread)
            camera_thread.start()

        # Keep the main thread running while cameras are being processed
        while any(thread.is_alive() for thread in camera_threads):
            time.sleep(1)  # Non-blocking wait, allowing for UI responsiveness

    except Exception as e:
        error_messages.append(str(e))  # Capture the error message
    finally:
        # Ensure all threads are signaled to stop
        for stop_event in stop_events:
            stop_event.set()

        # Ensure all windows are closed in the main thread
        for window in camera_windows:
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) >= 1:  # Check if window exists
                cv2.destroyWindow(window)

    # Check if there are any error messages
    if error_messages:
        # Join all error messages into a single string
        full_error_message = "\n".join(error_messages)
        return render(request, 'error.html', {'error_message': full_error_message})  # Render the error page with message

    return redirect('student_attendance_list')



def student_attendance_list(request):
    # Get the search query and date filter from the request
    search_query = request.GET.get('search', '')
    date_filter = request.GET.get('attendance_date', '')

    # Get all students
    students = Student.objects.all()

    # Filter students based on the search query
    if search_query:
        students = students.filter(name__icontains=search_query)

    # Prepare the attendance data
    student_attendance_data = []

    for student in students:
        # Get the attendance records for each student, filtering by attendance date if provided
        attendance_records = Attendance.objects.filter(roll_number=student)

        if date_filter:
            # Assuming date_filter is in the format YYYY-MM-DD
            attendance_records = attendance_records.filter(date=date_filter)

        attendance_records = attendance_records.order_by('date')
        
        student_attendance_data.append({
            'student': student,
            'attendance_records': attendance_records
        })

    context = {
        'student_attendance_data': student_attendance_data,
        'search_query': search_query,  # Pass the search query to the template
        'date_filter': date_filter       # Pass the date filter to the template
    }
    return render(request, 'student_attendance_list.html', context)



def home(request):
    return render(request, 'home.html')


# Custom user pass test for admin access
def is_admin(user):
    return user.is_superuser

@login_required
@user_passes_test(is_admin)
def student_list(request):
    students = Student.objects.all()
    return render(request, 'student_list.html', {'students': students})

@login_required
@user_passes_test(is_admin)
def student_detail(request, pk):
    student = get_object_or_404(Student, pk=pk)
    return render(request, 'student_detail.html', {'student': student})

@login_required
@user_passes_test(is_admin)
def student_authorize(request, pk):
    student = get_object_or_404(Student, pk=pk)
    
    if request.method == 'POST':
        authorized = request.POST.get('authorized', False)
        student.authorized = bool(authorized)
        student.save()
        return redirect('student-detail', pk=pk)
    
    return render(request, 'student_authorize.html', {'student': student})

# This views is for Deleting student
@login_required
@user_passes_test(is_admin)
def student_delete(request, pk):
    student = get_object_or_404(Student, pk=pk)
    
    if request.method == 'POST':
        student.delete()
        messages.success(request, 'Student deleted successfully.')
        return redirect('student-list')  # Redirect to the student list after deletion
    
    return render(request, 'student_delete_confirm.html', {'student': student})


# View function for user login
def user_login(request):
    # Check if the request method is POST, indicating a form submission
    if request.method == 'POST':
        # Retrieve username and password from the submitted form data
        username = request.POST.get('username')
        password = request.POST.get('password')

        # Authenticate the user using the provided credentials
        user = authenticate(request, username=username, password=password)

        # Check if the user was successfully authenticated
        if user is not None:
            # Log the user in by creating a session
            login(request, user)
            # Redirect the user to the student list page after successful login
            return redirect('home')  # Replace 'student-list' with your desired redirect URL after login
        else:
            # If authentication fails, display an error message
            messages.error(request, 'Invalid username or password.')

    # Render the login template for GET requests or if authentication fails
    return render(request, 'login.html')


# This is for user logout
def user_logout(request):
    logout(request)
    return redirect('login')  # Replace 'login' with your desired redirect URL after logout


# Function to handle the creation of a new camera configuration
@login_required
@user_passes_test(is_admin)
def camera_config_create(request):
    # Check if the request method is POST, indicating form submission
    if request.method == "POST":
        # Retrieve form data from the request
        name = request.POST.get('name')
        camera_source = request.POST.get('camera_source')
        threshold = request.POST.get('threshold')

        try:
            # Save the data to the database using the CameraConfiguration model
            CameraConfiguration.objects.create(
                name=name,
                camera_source=camera_source,
                threshold=threshold,
            )
            # Redirect to the list of camera configurations after successful creation
            return redirect('camera_config_list')

        except IntegrityError:
            # Handle the case where a configuration with the same name already exists
            messages.error(request, "A configuration with this name already exists.")
            # Render the form again to allow user to correct the error
            return render(request, 'camera_config_form.html')

    # Render the camera configuration form for GET requests
    return render(request, 'camera_config_form.html')


# READ: Function to list all camera configurations
@login_required
@user_passes_test(is_admin)
def camera_config_list(request):
    # Retrieve all CameraConfiguration objects from the database
    configs = CameraConfiguration.objects.all()
    # Render the list template with the retrieved configurations
    return render(request, 'camera_config_list.html', {'configs': configs})


# UPDATE: Function to edit an existing camera configuration
@login_required
@user_passes_test(is_admin)
def camera_config_update(request, pk):
    # Retrieve the specific configuration by primary key or return a 404 error if not found
    config = get_object_or_404(CameraConfiguration, pk=pk)

    # Check if the request method is POST, indicating form submission
    if request.method == "POST":
        # Update the configuration fields with data from the form
        config.name = request.POST.get('name')
        config.camera_source = request.POST.get('camera_source')
        config.threshold = request.POST.get('threshold')
        config.success_sound_path = request.POST.get('success_sound_path')

        # Save the changes to the database
        config.save()

        # Redirect to the list page after successful update
        return redirect('camera_config_list')

        # Render the configuration form with the current configuration data for GET requests
    return render(request, 'camera_config_form.html', {'config': config})


# DELETE: Function to delete a camera configuration
@login_required
@user_passes_test(is_admin)
def camera_config_delete(request, pk):
    # Retrieve the specific configuration by primary key or return a 404 error if not found
    config = get_object_or_404(CameraConfiguration, pk=pk)

    # Check if the request method is POST, indicating confirmation of deletion
    if request.method == "POST":
        # Delete the record from the database
        config.delete()
        # Redirect to the list of camera configurations after deletion
        return redirect('camera_config_list')

    # Render the delete confirmation template with the configuration data
    return render(request, 'camera_config_delete.html', {'config': config})


def attendance_form(request):
    return render(request, 'attendance_form.html')





import requests
from django.utils import timezone

# Function to send attendance data to manual project API
def send_attendance_to_manual_project(roll_number, subject, phase):
    try:
        # Define the URL for the manual project API endpoint
        manual_project_api_url = 'http://127.0.0.1:8000/facial/'

        # Prepare the data payload for the POST request
        data = {
            'roll_no': roll_number,
            'subject_name': subject,
            'phase': phase,
            'date': timezone.now().date(),  # Sending today's date
        }

        # Log the data being sent
        print(f"Sending data to manual project API: {data}")

        # Send POST request to manual project API
        response = requests.post(manual_project_api_url, data=data)

        # Log the response status and content
        if response.status_code == 200:
            print(f"Attendance for {roll_number} sent to manual project successfully.")
            print(f"Response from manual project API: {response.json()}")
        else:
            print(f"Failed to send attendance for {roll_number}. API response: {response.status_code}")
            print(f"Error details: {response.text}")
    except Exception as e:
        print(f"Error sending attendance data to manual project API: {e}")
