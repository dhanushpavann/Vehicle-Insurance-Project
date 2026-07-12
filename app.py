from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
from starlette.responses import HTMLResponse, RedirectResponse  
from uvicorn import run as app_run

from typing import Optional

#Importing pipelines and constants from other files
from src.constants import APP_HOST, APP_PORT
from src.pipline.prediction_pipeline import VehicleData, VehicleDataClassifier
from src.pipline.training_pipeline import TrainPipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load shared resources once when the app starts.
    """
    app.state.model_predictor = None
    app.state.model_ready = False
    app.state.model_load_error = None

    try:
        predictor = VehicleDataClassifier()
        predictor.model.loaded_model = predictor.model.load_model()
        app.state.model_predictor = predictor
        app.state.model_ready = True
    except Exception as e:
        app.state.model_load_error = str(e)
    yield


#Initialize FastAPI Application
app = FastAPI(lifespan=lifespan)

# Mount the 'static' directory for serving static files (like CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 template engine for rendering HTML templates
templates = Jinja2Templates(directory='templates')

# Allow all origins for Cross-Origin Resource Sharing (CORS)
origins = ["*"]

app.add_middleware(CORSMiddleware,
                   allow_origins = origins,
                   allow_credentials=True,
                   allow_methods = ["*"],
                   allow_headers = ["*"],
                   )


def reload_model(app_instance: FastAPI) -> None:
    """
    Refresh the in-memory model from S3 so the app uses the latest pushed model.
    """
    predictor = VehicleDataClassifier()
    predictor.model.loaded_model = predictor.model.load_model()
    app_instance.state.model_predictor = predictor
    app_instance.state.model_ready = True
    app_instance.state.model_load_error = None

class DataForm:
    """
    DataForm class is used to handle and process incoming form data
    This class defines the vehicle-related attributes from the form
    """

    def __init__(self,request: Request):
        self.request: Request = request
        self.Gender: Optional[int] = None
        self.Age: Optional[int] = None
        self.Driving_License: Optional[int] = None
        self.Region_Code: Optional[float] = None
        self.Previously_Insured: Optional[int] = None
        self.Annual_Premium: Optional[float] = None
        self.Policy_Sales_Channel: Optional[float] = None
        self.Vintage: Optional[int] = None
        self.Vehicle_Age_lt_1_Year: Optional[int] = None
        self.Vehicle_Age_gt_2_Years: Optional[int] = None
        self.Vehicle_Damage_Yes: Optional[int] = None

    # async def get_vehicle_data(self):
    #     """
    #     Method to retrieve and assign form data to class attributes.
    #     This method is asynchronous to handle form data fetching without blocking.
    #     """
    #     form = await self.request.form()
    #     self.Gender = form.get("Gender")
    #     self.Age = form.get("Age")
    #     self.Driving_License = form.get("Driving_License")
    #     self.Region_Code = form.get("Region_Code")
    #     self.Previously_Insured = form.get("Previously_Insured")
    #     self.Annual_Premium = form.get("Annual_Premium")
    #     self.Policy_Sales_Channel = form.get("Policy_Sales_Channel")
    #     self.Vintage = form.get("Vintage")
    #     self.Vehicle_Age_lt_1_Year = form.get("Vehicle_Age_lt_1_Year")
    #     self.Vehicle_Age_gt_2_Years = form.get("Vehicle_Age_gt_2_Years")
    #     self.Vehicle_Damage_Yes = form.get("Vehicle_Damage_Yes")

    async def get_vehicle_data(self):
        """
        Method to retrieve and assign form data to class attributes.
        This method is asynchronous to handle form data fetching without blocking.
        """
        form = await self.request.form()
        self.Gender = form.get("Gender")
        self.Age = int(form.get("Age"))
        self.Driving_License = int(form.get("Driving_License"))
        self.Region_Code = float(form.get("Region_Code"))
        self.Previously_Insured = int(form.get("Previously_Insured"))
        self.Annual_Premium = float(form.get("Annual_Premium"))
        self.Policy_Sales_Channel = float(form.get("Policy_Sales_Channel"))
        self.Vintage = int(form.get("Vintage"))
        self.Vehicle_Age_lt_1_Year = int(form.get("Vehicle_Age_lt_1_Year"))
        self.Vehicle_Age_gt_2_Years = int(form.get("Vehicle_Age_gt_2_Years"))
        self.Vehicle_Damage_Yes = int(form.get("Vehicle_Damage_Yes"))

    
# Route to render to the main page with the form
@app.get("/", tags = ['authentication'])
async def index(request: Request):
    """
    Renders the main HTML form page for vehicle input
    """
    
    return templates.TemplateResponse(
    request=request,
    name="vehicledata.html",
    context={"context": "Rendering"}
)


@app.get("/health")
async def health(request: Request):
    """
    Report whether the model is loaded and ready for predictions.
    """
    return {
        "status": "ok" if request.app.state.model_ready else "degraded",
        "model_ready": request.app.state.model_ready,
        "model_loaded": request.app.state.model_predictor is not None,
        "error": request.app.state.model_load_error,
    }

# Route to trigger the model training process
@app.post("/train")
async def trainRouteClient():
    """
    Endpoint to initiate the model training pipeline. 
    """
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        reload_model(app)
        return Response("Training Succesfull!!")
    
    except Exception as e:
        return Response(f"Error occurred while training: {e}")
    
# Route to handle Form submission and handle prediction
@app.post("/")
async def predictRouteClient(request: Request):
    """
    Endpoint to receive form data, process it and make prediction.
    """
    try:
        if not request.app.state.model_ready:
            return {"status": False, "error": "Model is not loaded yet.", "health": await health(request)}

        form = DataForm(request)
        await form.get_vehicle_data()

        vehicle_data = VehicleData(
            
            Gender= form.Gender,
            Age = form.Age,
            Driving_License = form.Driving_License,
            Region_Code = form.Region_Code,
            Previously_Insured = form.Previously_Insured,
            Annual_Premium = form.Annual_Premium,
            Policy_Sales_Channel = form.Policy_Sales_Channel,
            Vintage = form.Vintage,
            Vehicle_Age_lt_1_Year = form.Vehicle_Age_lt_1_Year,
            Vehicle_Age_gt_2_Years = form.Vehicle_Age_gt_2_Years,
            Vehicle_Damage_Yes = form.Vehicle_Damage_Yes
            )
        # Convert the form data into a DataFrame
        vehicle_df = vehicle_data.get_vehicle_input_data_frame()

        # Reuse the preloaded predictor instead of creating a new one per request
        model_predictor = request.app.state.model_predictor

        # Make a prediction and retrive the result
        value = model_predictor.predict(dataframe=vehicle_df)[0]

        # Interpret the prediction result as 'Response-Yes' or 'Response-No'
        status = "Response-Yes" if value == 1 else "Response-No"

        # Render the same HTML page with  prediction result
        return templates.TemplateResponse(
        request=request,
        name="vehicledata.html",
        context={"context": status},
    )
    except Exception as e:
        return {"status": False,"error": f"{e}"}
    

#Main entry point to start the FastAPI server
if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)
    
