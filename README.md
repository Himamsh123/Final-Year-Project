# AI Powered Web Application for retinal disease detection using transformer models
 
A full-stack web application where users can log in and upload retinal images to predict Glaucoma and Diabetic Retinopathy (DR) using Vision Transformer, Swin Transformer, Twin Transformer. The system also includes a RAG-based chatbot for medical queries and a feature to generate PDF reports with user and prediction details.

## Project Overview 
This project integrates the best performed model into the backend of the application to build a medical diagnostic tool. It allows:

• Authentication: Signup, login, forgot password along with google signup and login  
• Image Upload: Upload retinal images through a responsive UI  
• Prediction: Transformer model will predict the output  
• Result Display: Table with results for DR and Glaucoma will be displayed  
• PDF Report: Generate downloadable report with patient info, predictions and suggestions   
• Chatbot: Ask questions to a Gemini-based assistant built using RAG  

## Set-up instructions
1. Clone the repository
git clone https://github.com/GeethaDeepika/fsd_application.git  
cd fsd_application

2. Install the required packages
cd frontend
npm install

3. Add the saved models to the main directory 

4. Update the .env file in both frontend and backend
