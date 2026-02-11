#!/usr/bin/env python3
"""
MitoReach: Genome-Based Mitochondrial Reach Scanner
Analyzes mtDNA genomes, identifies TC motifs, and runs MitoReach structural reach predictions.

Usage:
    python3 run_mitoreach.py --genome <genbank_file> --gene <name> --position <pos> [options]
"""

import sys
import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from genome_parser import GenomeParser
from tc_motif_scanner import TCMotifScanner


def find_optimal_spacer_for_target(sequence: str, target_offset: int, strand: str, spacer_length: int = 15) -> Tuple[Optional[str], int, int]:
    """
    Find optimal spacer where target C falls in editable window.
    
    Architecture-Specific Windows (Split-DddA):
    - TS (Template Strand/Bottom): Positions 3-8 (N-terminal DddA reach)
    - NTS (Non-Template Strand/Top): Positions 8-13 (C-terminal DddA reach)
    
    Args:
        sequence: DNA sequence
        target_offset: Offset of target in sequence
        strand: Target strand ('NTS' or 'TS')
        spacer_length: Desired spacer length
    
    Returns:
        Tuple of (spacer_seq, target_pos_in_spacer, spacer_start_offset)
    """
    # Define preferred positions based on strand architecture
    # Prioritize 'Safe Optimal' (Position 4 for TS, Position 9 for NTS)
    if strand == 'TS':
        # TS Window: 3-8. Best: 4 (Safe), 5, 3
        preferred_positions = [4, 5, 3, 6, 7, 8]
    else:
        # NTS Window: 8-13. Best: 9, 10, 8, 11, 12, 13
        preferred_positions = [9, 10, 8, 11, 12, 13]
    
    for pos_in_spacer in preferred_positions:
        # Calculate start of spacer if target is at 'pos_in_spacer'
        spacer_start = target_offset - (pos_in_spacer - 1)
        spacer_end = spacer_start + spacer_length
        
        # Check bounds
        if spacer_start >= 0 and spacer_end <= len(sequence):
            spacer_seq = sequence[spacer_start:spacer_end]
            return spacer_seq, pos_in_spacer, spacer_start

    # Fallback: simple centering if no optimal found
    return extract_spacer_for_position(sequence, target_offset, spacer_length) + (target_offset - (spacer_length // 2) + 1,)


def extract_spacer_for_position(sequence: str, target_offset: int, spacer_length: int = 16) -> tuple:
    """
    Extract spacer sequence centered on target position (Fallback).
    """
    # Calculate spacer boundaries
    half_spacer = spacer_length // 2
    spacer_start = max(0, target_offset - half_spacer + 1)
    spacer_end = min(len(sequence), target_offset + half_spacer + 1)
    
    # Extract spacer
    spacer_seq = sequence[spacer_start:spacer_end]
    
    # Pad if needed
    if len(spacer_seq) < spacer_length:
        if spacer_start == 0:
            spacer_seq = 'N' * (spacer_length - len(spacer_seq)) + spacer_seq
        else:
            spacer_seq = spacer_seq + 'N' * (spacer_length - len(spacer_seq))
            
    return spacer_seq, spacer_start


def run_mitoreach_core_analysis(spacer_seq: str, target_pos_in_spacer: int, output_file: str):
    """
    Run MitoReach core analysis for a specific spacer.
    
    Args:
        spacer_seq: Spacer DNA sequence
        target_pos_in_spacer: Position of target within spacer (1-indexed)
        output_file: Output PNG file
    
    Returns:
        Dictionary with MitoReach results
    """
    # Build temporary command
    import subprocess
    cmd = [
        "python3", "run_mitoreach_core.py",
        "--spacer", spacer_seq,
        "--output", output_file
    ]
    
    if target_pos_in_spacer:
        cmd.extend(["--target_pos", str(target_pos_in_spacer)])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        
        # Parse output for editability info
        output_lines = result.stdout.split('\n')
        
        editable_status = None
        recommended_pos = None
        
        for line in output_lines:
            if "[Success]" in line:
                editable_status = "editable"
            elif "[Recommendation]" in line:
                editable_status = "not_editable"
                # Extract recommended position if present
                if "Position" in line:
                    import re
                    match = re.search(r'Position (\d+)', line)
                    if match:
                        recommended_pos = int(match.group(1))
        
        return {
            'spacer': spacer_seq,
            'target_position': target_pos_in_spacer,
            'editable': editable_status == "editable",
            'recommended_position': recommended_pos,
            'output_file': output_file,
            'stdout': result.stdout
        }
        
    except Exception as e:
        print(f"Error running MitoReach: {e}")
        return None


def plot_alignment_map(sequence: str, spacer_start: int, spacer_length: int, 
                      target_offset: int, strand: str, output_file: str,
                      window_start_genomic: int):
    """
    Generate a graphical alignment map.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Define window (show some context around spacer)
    context_pad = 10
    view_start = max(0, spacer_start - context_pad)
    view_end = min(len(sequence), spacer_start + spacer_length + context_pad)
    view_seq = sequence[view_start:view_end]
    
    # Setup coordinates
    x_coords = np.arange(view_start, view_end)
    y_base = 0.5
    
    # Draw Sequence Grid
    for i, (idx, base) in enumerate(zip(x_coords, view_seq)):
        # Base text
        color = 'black'
        weight = 'normal'
        
        # Spacer highlighting
        if spacer_start <= idx < spacer_start + spacer_length:
            bg_color = 'lightyellow'
            weight = 'bold'
            
            # Editable Window Shading (Architecture specific)
            rel_pos = idx - spacer_start + 1  # 1-indexed in spacer
            editable = False
            
            if strand == 'TS':
                if 3 <= rel_pos <= 8: editable = True
            else:
                if 8 <= rel_pos <= 11: editable = True
                
            if editable:
                bg_color = 'lightgreen'
            
            rect = plt.Rectangle((i - 0.5, 0), 1, 1, facecolor=bg_color, edgecolor='none', alpha=0.5)
            ax.add_patch(rect)
        
        # Highlight Target
        if idx == target_offset:
            circle = plt.Circle((i, 0.5), 0.4, color='red', alpha=0.3)
            ax.add_patch(circle)
            color = 'darkred'
            weight = 'bold'
            
        ax.text(i, 0.5, base, ha='center', va='center', fontsize=12, 
                family='monospace', weight=weight, color=color)
        
        # Genomic position annotation (every 5)
        genomic_pos = window_start_genomic + idx
        if genomic_pos % 5 == 0:
            ax.text(i, -0.2, str(genomic_pos), ha='center', va='top', fontsize=8, rotation=90)

    # Spacer Bracket
    spacer_view_start = spacer_start - view_start
    spacer_view_end = spacer_view_start + spacer_length
    
    ax.plot([spacer_view_start - 0.5, spacer_view_end - 0.5], [1.1, 1.1], color='blue', lw=2)
    ax.text(spacer_view_start - 0.5, 1.2, f"Spacer ({spacer_length}bp)", 
            ha='left', va='bottom', color='blue', fontsize=10)

    # Editable Window Bracket
    if strand == 'TS':
        win_start_rel = 3
        win_end_rel = 8
        label = "Editable (TS: 3-8)"
    else:
        win_start_rel = 8
        win_end_rel = 11
        label = "Editable (NTS: 8-11)"
        
    win_start_idx = spacer_view_start + win_start_rel - 1
    win_end_idx = spacer_view_start + win_end_rel
    
    ax.plot([win_start_idx - 0.5, win_end_idx - 0.5], [-0.5, -0.5], color='green', lw=2)
    ax.text((win_start_idx + win_end_idx)/2 - 0.5, -0.6, label, 
            ha='center', va='top', color='green', fontsize=10)

    # Target Annotation
    target_idx_view = target_offset - view_start
    ax.annotate('Target', xy=(target_idx_view, 0.8), xytext=(target_idx_view, 1.5),
                arrowprops=dict(facecolor='red', shrink=0.05), ha='center', color='red')

    # Styling
    ax.set_ylim(-1.5, 2.5)
    ax.set_xlim(-0.5, len(view_seq) - 0.5)
    ax.axis('off')
    ax.set_title(f"Genomic Alignment Map (Target {strand})", pad=20)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()



def generate_report(genome_file: str, gene_name: str, target_position: int,
                   window_start: int, window_end: int, sequence: str,
                   all_tc_motifs: dict, edit_trace_result: dict,
                   output_dir: str):
    """
    Generate comprehensive markdown report.
    """
    report_path = os.path.join(output_dir, "genome_edit_trace_report.md")
    
    with open(report_path, 'w') as f:
        f.write(f"# MitoReach Analysis Report\n\n")
        
        # Methodology Note
        f.write(f"## Methodology: Mitochondrial Reach Simulation\n\n")
        f.write(f"> **How is reach analyzed without a specific structure?**\n")
        f.write(f"> MitoReach uses a physics-based simulation derived from the `9jo8` template structure. ")
        f.write(f"The Split-DddA domain is virtually rotated and translated along the DNA helical axis relative to the TALE anchor. ")
        f.write(f"This allows us to predict the 'reach' of the catalytic domain to any specific nucleotide position without needing a new PDB file for every sequence.\n\n")
        
        # Input Summary
        f.write(f"## Input Summary\n\n")
        f.write(f"- **Genome**: `{genome_file}`\n")
        if gene_name:
            f.write(f"- **Gene**: {gene_name}\n")
        f.write(f"- **Target Position**: {target_position}\n")
        f.write(f"- **Scanning Window**: [{window_start}, {window_end}] (\u00b150bp)\n")
        
        # TC Motif Summary
        f.write(f"## TC Motif Scan Results\n\n")
        nts_count = len(all_tc_motifs['NTS'])
        ts_count = len(all_tc_motifs['TS'])
        f.write(f"Found **{nts_count + ts_count} TC motifs** in scanning window.\n\n")
        
        # MitoReach Results
        f.write(f"## MitoReach Results\n\n")
        
        if edit_trace_result:
            spacer = edit_trace_result.get('spacer', '')
            t_pos = edit_trace_result.get('target_position', 0)
            
            f.write(f"### Optimal Protospacer Configuration\n\n")
            f.write(f"**Selected Protospacer**: `{spacer}`\n")
            f.write(f"**Target Position in Spacer**: {t_pos}\n")
            
            if edit_trace_result['editable']:
                f.write(f"\n> [!NOTE]\n")
                f.write(f"> **Target Position {target_position} is EDITABLE** with this configuration.\n\n")
            else:
                f.write(f"\n> [!WARNING]\n")
                f.write(f"> **Target Position {target_position} is NOT EDITABLE** even with optimization.\n\n")
                if edit_trace_result['recommended_position']:
                    f.write(f"**Recommendation**: Consider shifting to position {edit_trace_result['recommended_position']}\n\n")
            
            # Embed plot
            if os.path.exists(edit_trace_result['output_file']):
                f.write(f"### MitoReach Prediction Plot\n\n")
                f.write(f"![MitoReach Analysis]({os.path.basename(edit_trace_result['output_file'])})\n\n")

        else:
            f.write(f"> [!CAUTION]\n")
            f.write(f"> MitoReach analysis could not be performed\n\n")
        
        # Sequence Details
        f.write(f"## Sequence Details\n\n")
        f.write(f"```\n")
        f.write(f"Window: {window_start}-{window_end}\n")
        f.write(f"Length: {len(sequence)} bp\n\n")
        # Show sequence with line breaks
        for i in range(0, len(sequence), 60):
            pos = window_start + i
            seq_chunk = sequence[i:i+60]
            f.write(f"{pos:6d}: {seq_chunk}\n")
        f.write(f"```\n\n")
    
    print(f"\n\u2705 Report saved to: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(
        description="MitoReach: Genome-Based Mitochondrial Reach Scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all genes in genome
  python3 run_mitoreach.py --genome "human mtDNA .gb" --list-genes
  
  # Analyze specific position in ND5
  python3 run_mitoreach.py --genome "human mtDNA .gb" --gene ND5 --position 13514
  
  # Analyze with custom scanning window
  python3 run_mitoreach.py --genome "human mtDNA .gb" --position 13514 --scan-window 100
        """
    )
    
    parser.add_argument("--genome", required=True, help="Path to GenBank file")
    parser.add_argument("--gene", help="Gene name (for validation and context)")
    parser.add_argument("--position", type=int, help="Target genomic position (1-indexed)")
    parser.add_argument("--scan-window", type=int, default=50, help="Scanning window size (\u00b1bp, default: 50)")
    parser.add_argument("--spacer-length", type=int, default=15, help="Spacer length for MitoReach (default: 15)")
    parser.add_argument("--output-dir", default="genome_edit_trace_results", help="Output directory")
    parser.add_argument("--list-genes", action="store_true", help="List all genes and exit")
    
    args = parser.parse_args()
    
    # Load genome
    print(f"Loading genome from: {args.genome}")
    genome = GenomeParser(args.genome)
    
    # List genes if requested
    if args.list_genes:
        print("\n=== Genes in Genome ===")
        for gene in genome.list_genes():
            print(f"{gene['name']:20s} {gene['start']:6d}-{gene['end']:6d}  "
                  f"({gene['end']-gene['start']+1:5d} bp)  {gene['product']}")
        return
    
    # Validate required arguments
    if not args.position:
        print("Error: --position is required (use --list-genes to see gene coordinates)")
        sys.exit(1)
    
    # Validate position
    validation = genome.validate_position(args.position, args.gene)
    print(f"\n{validation['message']}")
    
    if not validation['valid']:
        sys.exit(1)
    
    # Get scanning window
    window_start, window_end, sequence = genome.get_scanning_window(args.position, args.scan_window)
    print(f"\nScanning window: [{window_start}, {window_end}]")
    print(f"Extracted sequence: {len(sequence)} bp")
    
    # Scan for TC motifs
    print(f"\nScanning for TC motifs...")
    scanner = TCMotifScanner()
    all_tc_motifs = scanner.find_tc_motifs(sequence, window_start)
    
    print(scanner.summarize_motifs(all_tc_motifs))
    
    # Check if target position has TC motif
    has_tc, target_motif = scanner.check_position_has_tc(sequence, window_start, args.position)
    
    if not has_tc:
        print(f"\u26a0\ufe0f  WARNING: Position {args.position} does not contain a TC motif!")
        print(f"   DddA requires a 5'-TC-3' motif for editing.\n")
        
        # Find nearest TC
        nearest_tcs = scanner.find_nearest_tc_motifs(all_tc_motifs, args.position, args.scan_window)
        if nearest_tcs:
            print(f"   Nearest TC motifs:")
            for i, motif in enumerate(nearest_tcs[:5], 1):
                print(f"     {i}. Position {motif['position']} ({motif['strand']}) - "
                      f"{motif['distance']} bp {motif['direction']}")
        
        print(f"\nCannot run MitoReach without a TC motif at target position.")
        sys.exit(1)
    
    print(f"\u2705 Target position {args.position} has TC motif on {target_motif['strand']}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract spacer for MitoReach
    target_offset = args.position - window_start
    spacer_seq, target_in_spacer, spacer_start_offset = find_optimal_spacer_for_target(
        sequence, target_offset, target_motif['strand'], args.spacer_length
    )
    
    # Bystander Mitigation Logic
    print(f"\nChecking for bystanders in {args.spacer_length}bp window...")
    motifs = scanner.find_tc_motifs(spacer_seq)
    total_tcs = len(motifs['NTS']) + len(motifs['TS'])
    
    if total_tcs > 1:
        print(f"\u26a0\ufe0f Bystanders found! Attempting to reduce length for specificity...")
        for length in range(args.spacer_length - 1, 9, -1):
            # Try to keep the same start position and target position
            new_spacer = spacer_seq[:length]
            new_motifs = scanner.find_tc_motifs(new_spacer)
            new_total = len(new_motifs['NTS']) + len(new_motifs['TS'])
            
            if new_total == 1:
                print(f"\u2705 Found bystander-free window at {length}bp.")
                spacer_seq = new_spacer
                args.spacer_length = length
                break
    else:
        print(f"\u2705 15bp window is clean.")

    # Run MitoReach
    print(f"\nRunning MitoReach analysis with optimized spacer...")
    print(f"Optimal Protospacer: {spacer_seq}")
    print(f"Target position in spacer: {target_in_spacer}")
    
    output_plot = os.path.join(args.output_dir, f"target_position_{args.position}_mitoreach.png")
    
    edit_trace_result = run_mitoreach_core_analysis(spacer_seq, target_in_spacer, output_plot)
    
    # Generate Alignment Map
    print(f"Generating alignment map...")
    alignment_map_file = os.path.join(args.output_dir, "alignment_map.png")
    plot_alignment_map(
        sequence, spacer_start_offset, args.spacer_length,
        target_offset, target_motif['strand'], alignment_map_file,
        window_start
    )
    
    # Generate report
    print(f"\nGenerating report...")
    report_path = generate_report(
        args.genome,
        args.gene,
        args.position,
        window_start,
        window_end,
        sequence,
        all_tc_motifs,
        edit_trace_result,
        args.output_dir
    )
    
    # Append alignment map to report
    with open(report_path, 'a') as f:
        f.write(f"\n### Genomic Alignment & Editable Window\n\n")
        f.write(f"![Genomic Alignment Map]({os.path.basename(alignment_map_file)})\n\n")
    
    # Save results JSON
    results_json = os.path.join(args.output_dir, "results.json")
    results_data = {
        'genome_file': args.genome,
        'gene': args.gene,
        'target_position': args.position,
        'scanning_window': {'start': window_start, 'end': window_end},
        'tc_motifs': {
            'NTS': [{'position': m['position'], 'context': m['context']} for m in all_tc_motifs['NTS']],
            'TS': [{'position': m['position'], 'context': m['context']} for m in all_tc_motifs['TS']]
        },
        'edit_trace': edit_trace_result
    }
    
    with open(results_json, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\u2705 Results saved to: {results_json}")
    print(f"\n\u2728 Analysis complete! Check {args.output_dir}/ for results.")


if __name__ == '__main__':
    main()
