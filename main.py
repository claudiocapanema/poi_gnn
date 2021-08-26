import sys
import ast
import os
import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from job.poi_categorization_job import PoiCategorizationJob
from job.poi_categorization_baselines_job import PoiCategorizationBaselinesJob
from job.poi_categorization_performance_graphics_job import PoiCategorizationPerformanceGraphicsJob
from job.matrix_generation_for_poi_categorization_job import MatrixGenerationForPoiCategorizationJob
from job.poi_transactions_analysis_job import PoiTransactionsAnalysisJob
from job.hmrm_job import HmrmBaseline
from job.poi_categorization_baseline_gpr_job import PoiCategorizationBaselineGPRJob
from foundation.configuration.input import Input

def start_input(args):
    Input().set_inputs(args)


def start_job(args):

    start_input(args)
    job_name = Input.get_instance().inputs['job']
    print(job_name)
    if job_name == "categorization":
        job = PoiCategorizationJob()
    elif job_name == "categorization_baselines":
        job = PoiCategorizationBaselinesJob()
    elif job_name == "categorization_performance_graphics":
        job = PoiCategorizationPerformanceGraphicsJob()
    elif job_name == "matrix_generation_for_poi_categorization":
        job = MatrixGenerationForPoiCategorizationJob()
    elif job_name == "hmrm_baseline":
        job = HmrmBaseline()
    elif job_name == "poi_categorization_baseline_gpr":
        job = PoiCategorizationBaselineGPRJob()
    elif job_name == "poi_transactions":
        job = PoiTransactionsAnalysisJob()

    job.start()

if __name__ == "__main__":
    try:

        args = ast.literal_eval(sys.argv[1])
        start_job(args)

    except Exception as e:
        raise e