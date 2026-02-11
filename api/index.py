from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import base64
import subprocess
from typing import List, Dict

# Import your core logic
import run_mitoreach
from genome_parser import GenomeParser
from tc_motif_scanner import TCMotifScanner

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

class AnalysisRequest(BaseModel):
    genome_key: str
    target_pos: int
    spacer_len: int = 15
    linker_type: str = "short"

@app.get("/api/genomes")
def get_genomes():
    return {
        "Zebrafish (Danio rerio)": "zebrafish_mtDNA.gb",
        "Human (Homo sapiens)": "human mtDNA .gb",
        "Mouse (Mus musculus)": "mouse mtdna.gb"
    }

@app.post("/api/analyze")
async def analyze(request: AnalysisRequest):
    genome_file = request.genome_key # We expect the filename here
    genome_path = os.path.join(DATA_DIR, genome_file)
    
    if not os.path.exists(genome_path):
        raise HTTPException(status_code=404, detail=f"Genome file {genome_file} not found")

    # 1. Load Genome
    genome = GenomeParser(genome_path)
    validation = genome.validate_position(request.target_pos)
    
    if not validation['valid']:
        raise HTTPException(status_code=400, detail=validation['message'])
        
    # 2. Extract Window
    window_start, window_end, sequence = genome.get_scanning_window(request.target_pos, 50)
    
    # 3. Check for TC
    scanner = TCMotifScanner()
    has_tc, target_motif = scanner.check_position_has_tc(sequence, window_start, request.target_pos)
    
    if not has_tc:
        return {"status": "no_tc", "message": f"Position {request.target_pos} has no TC motif."}

    # 4. Find Spacer
    target_offset = request.target_pos - window_start
    spacer_seq, target_in_spacer, _ = run_mitoreach.find_optimal_spacer_for_target(
        sequence, target_offset, target_motif['strand'], request.spacer_len
    )
    
    # 5. Run Core Simulation
    # Use a temporary file for the plot
    output_plot = f"/tmp/prediction_{request.target_pos}.png"
    core_script = os.path.join(os.path.dirname(__file__), "run_mitoreach_core.py")
    
    cmd = [
        "python3", core_script,
        "--spacer", spacer_seq,
        "--target_pos", str(target_in_spacer),
        "--linker_type", request.linker_type,
        "--output", output_plot,
        "--export_json"
    ]
    
    process = subprocess.run(cmd, capture_output=True, text=True, cwd=BASE_DIR)
    
    if process.returncode != 0:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {process.stderr}")
        
    # 6. Prepare Response
    json_file = output_plot.replace(".png", "_analysis.json")
    results_data = {}
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            results_data = json.load(f)
            
    # Convert image to base64 to send to frontend
    img_base64 = ""
    if os.path.exists(output_plot):
        with open(output_plot, "rb") as image_file:
            img_base64 = base64.b64encode(image_file.read()).decode('utf-8')
            
    return {
        "status": "success",
        "results": results_data,
        "plot": img_base64,
        "spacer": spacer_seq,
        "motif": target_motif,
        "score": results_data.get('target_recommendation', {}).get('best_score', 0)
    }
