from clearml import Task, Model
task = Task.init(project_name="Hazard_Detection", task_name="test_model_fetch")
models = Model.query_models(project_name="Hazard_Detection")
if models:
    m = models[-1]
    print(dir(m))
