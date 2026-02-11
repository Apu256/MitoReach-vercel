#!/usr/bin/env python3
"""
MitoReach Core: Structural Reach Prediction
Usage:
    python3 run_mitoreach_core.py --spacer <sequence> --output <filename.png>
"""

import sys
import numpy as np
import subprocess
import re
import os
import argparse
import csv
import json
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Enable non-interactive backend for PDF/SVG

# Constants (Template specific - 9jo8)
# Constants
C_axis = np.array([168.962, 167.737, 181.423])
V_axis = np.array([-0.019, -0.234, 0.972])
V_axis = V_axis / np.linalg.norm(V_axis)

def rotation_matrix_from_axis_angle(axis, angle_deg):
    angle_rad = np.radians(angle_deg)
    ux, uy, uz = axis
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [c + ux**2 * (1-c), ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
        [uy*ux*(1-c) + uz*s, c + uy**2 * (1-c), uy*uz*(1-c) - ux*s],
        [uz*ux*(1-c) - uy*s, uz*uy*(1-c) + ux*s, c + uz**2 * (1-c)]
    ])

def parse_pdb_atoms(pdb_file):
    tale_atoms = []
    other_lines = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                chain_id = line[21]
                if chain_id == 'A':
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    tale_atoms.append({
                        'line': line,
                        'coords': np.array([x, y, z])
                    })
                else:
                    other_lines.append(line)
            else:
                other_lines.append(line)
    return tale_atoms, other_lines
def run_sasd(pdb_file, chain1, res1, chain2, res2):
    try:
        sasd_script = os.path.join(os.path.dirname(__file__), "sasd_calculator.py")
        cmd = [
            "python3", sasd_script, pdb_file,
            "--start_chain", str(chain1),
            "--start_res", str(res1),
            "--end_chain", str(chain2),
            "--end_res", str(res2)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout
        
        match = re.search(r"Calculated SASD:\s*([\d\.]+)\s*Angstroms", output)
        if match: return float(match.group(1))
        
        match = re.search(r"SASD between .* and .*: ([\d\.]+) A", output)
        if match: return float(match.group(1))
            
        return None
    except Exception:
        return None

def calculate_strand_score(dist_n, dist_c, threshold, base, context_motif, pos, strand='NTS', optimal=40.0):
    """
    Calculate strand-specific editing score (0-100).
    Factors:
    1. Reach (Distance)
    2. Base Identity (Must be 'C')
    3. Context (Bonus for 'TC')
    4. Window Favorability (NTS: 8-13, TS: 3-8)
    """
    # 1. Base Identity
    if base.upper() != 'C':
        return 0.0
        
    # 2. Reach Score
    max_dist = max(dist_n, dist_c)
    reach_score = 0.0
    if max_dist <= optimal:
        reach_score = 100.0
    elif max_dist >= threshold:
        reach_score = 0.0
    else:
        decay_range = threshold - optimal
        penalty = (max_dist - optimal) / decay_range
        reach_score = 100.0 * (1.0 - penalty)
        
    # 3. Context Multiplier
    # DddA strongly prefers 5'-TC.
    final_score = reach_score
    if context_motif and context_motif.upper() != 'TC':
        final_score *= 0.5 # 50% penalty for non-TC context
            
    # 4. Window Favorability (Experimental Reality)
    # Top (NTS) favors downstream (8-13)
    # Bottom (TS) favors upstream (3-8)
    if strand == 'NTS':
        if not (8 <= pos <= 13):
            final_score *= 0.1 # Strong penalty for structurally close but biologically locked positions (like Pos 3)
    elif strand == 'TS':
        if not (3 <= pos <= 8):
            final_score *= 0.1
            
    return final_score

def main():
    parser = argparse.ArgumentParser(description="MitoReach Core: Structural Reach Prediction")
    parser.add_argument("--spacer", required=True, help="Spacer DNA sequence (NTS, 5'-3')")
    parser.add_argument("--output", default="edit_trace_prediction.png", help="Output PNG file")
    parser.add_argument("--target_pos", type=int, help="Optional: Specific target position to check")
    parser.add_argument("--linker_type", choices=['short', 'long'], default='short', 
                        help="Linker configuration: 'short' (~16aa, 70A limit) or 'long' (~63aa, 120A limit)")
    parser.add_argument("--export_csv", action='store_true', help="Export raw SASD data to CSV")
    parser.add_argument("--export_json", action='store_true', help="Export complete analysis to JSON")
    parser.add_argument("--graph_format", choices=['png', 'svg', 'pdf', 'all'], default='png',
                        help="Output format for graphs (default: png)")
    args = parser.parse_args()
    
    # Configuration
    base_pdb = "data/9jo8.chimera.pdb"
    if not os.path.exists(base_pdb):
        print(f"Error: Template {base_pdb} not found. Ensure it is in the 'data' directory.")
        return

    # Linker Parameters
    if args.linker_type == 'long':
        reach_threshold = 120.0  # Extended reach for 63aa
        optimal_dist = 60.0      # Relaxed optimal range
        print(f"Configuration: Long Linker (63aa) | Threshold: {reach_threshold} Å")
    else:
        reach_threshold = 70.0   # Short reach for 16aa
        optimal_dist = 40.0
        print(f"Configuration: Short Linker (16aa) | Threshold: {reach_threshold} Å")

    # Sequence Logic
    spacer_seq_nts = args.spacer.upper()
    complement_map = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    spacer_seq_ts = ''.join([complement_map.get(b, 'N') for b in spacer_seq_nts])
    
    cytosines_nts = [(i+1, 'NTS') for i, base in enumerate(spacer_seq_nts) if base == 'C']
    cytosines_ts = [(i+1, 'TS') for i, base in enumerate(spacer_seq_ts) if base == 'C']
    
    print(f"Spacer NTS: {spacer_seq_nts}")
    print(f"Spacer TS : {spacer_seq_ts}")
    
    # Simulation
    tale_atoms, other_lines = parse_pdb_atoms(base_pdb)
    positions = range(1, len(spacer_seq_nts) + 1)
    shifts = [p - 4 for p in positions]
    results = []
    
    print("Running Simulation...")
    for pos, shift in zip(positions, shifts):
        angle = shift * 36.0
        translation = shift * 3.4 * V_axis
        R = rotation_matrix_from_axis_angle(V_axis, angle)
        
        shifted_lines = []
        for atom in tale_atoms:
            rel_pos = atom['coords'] - C_axis
            rot_pos = np.dot(R, rel_pos)
            new_pos = rot_pos + C_axis + translation
            
            x_str = f"{new_pos[0]:8.3f}"
            y_str = f"{new_pos[1]:8.3f}"
            z_str = f"{new_pos[2]:8.3f}"
            
            orig_line = atom['line']
            new_line = orig_line[:30] + x_str + y_str + z_str + orig_line[54:]
            shifted_lines.append(new_line)
            
        temp_pdb = f"temp_gen_{shift}.pdb"
        with open(temp_pdb, 'w') as f:
            for line in other_lines:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    f.write(line)
                elif line.startswith("END"): pass
                else: f.write(line)
            for line in shifted_lines:
                f.write(line)
            f.write("END\n")
            
        sasd_n = run_sasd(temp_pdb, 'A', 641, 'D', 1290)
        sasd_c = run_sasd(temp_pdb, 'A', 641, 'D', 1398)
        
        if os.path.exists(temp_pdb): os.remove(temp_pdb)
        
        base_n = spacer_seq_nts[pos-1] if pos <= len(spacer_seq_nts) else 'N'
        base_t = spacer_seq_ts[pos-1] if pos <= len(spacer_seq_ts) else 'N'
        
        # Determine Context
        # NTS Context (5'-X C)
        ctx_n_motif = None
        if pos > 1:
            prev_base = spacer_seq_nts[pos-2]
            ctx_n_motif = prev_base + base_n
            
        # TS Context (5'-X C)
        ctx_t_motif = None
        if pos < len(spacer_seq_ts):
             prev_base_ts = spacer_seq_ts[pos] # pos is 1-based, index is pos-1. Next index is pos.
             ctx_t_motif = prev_base_ts + base_t
             
        score_n = 0.0
        score_t = 0.0
        
        if sasd_n is not None and sasd_c is not None:
            dn_calib = sasd_n * 1.15
            dc_calib = sasd_c * 1.15
            
            score_n = calculate_strand_score(dn_calib, dc_calib, reach_threshold, base_n, ctx_n_motif, pos, 'NTS', optimal_dist)
            score_t = calculate_strand_score(dn_calib, dc_calib, reach_threshold, base_t, ctx_t_motif, pos, 'TS', optimal_dist)
            
        results.append((pos, base_n, base_t, sasd_n, sasd_c, score_n, score_t, ctx_n_motif, ctx_t_motif))
        print(f"  Pos {pos}: NTS_Score={score_n:.1f} ({ctx_n_motif}), TS_Score={score_t:.1f} ({ctx_t_motif})")
        # Machine readable output for optimizer (1.15 calibrated values)
        if sasd_n and sasd_c:
             print(f"RAW_DATA|{pos}|{sasd_n*1.15:.2f}|{sasd_c*1.15:.2f}|{base_n}|{ctx_n_motif}|{base_t}|{ctx_t_motif}")

    # Plot
    valid_results = [r for r in results if r[3] is not None and r[4] is not None]
    if not valid_results:
        print("No valid results to plot.")
        return

    valid_pos = [r[0] for r in valid_results]
    sasd_n_vals = [r[3] * 1.15 for r in valid_results]
    sasd_c_vals = [r[4] * 1.15 for r in valid_results]
    score_n_vals = [r[5] for r in valid_results]
    score_t_vals = [r[6] for r in valid_results]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # --- Distances Only ---
    # Editable (<Threshold)
    for i, (dn, dc) in enumerate(zip(sasd_n_vals, sasd_c_vals)):
        if dn <= reach_threshold and dc <= reach_threshold:
            ax.axvspan(valid_pos[i] - 0.5, valid_pos[i] + 0.5, color='lightgreen', alpha=0.1, zorder=1)
            
    # Biological Sweet Spots (Experimental Data) removed for minimal view.
            
    total_len = len(spacer_seq_nts)
    ax.axvspan(0.5, 2.5, color='gray', alpha=0.3, label='Steric Hindrance', zorder=1)
    ax.axvspan(total_len - 1.5, total_len + 0.5, color='gray', alpha=0.3, zorder=1)
    ax.axhline(y=reach_threshold, color='k', linestyle=':', label=f'Reach Limit ({reach_threshold} A)', zorder=2)
    
    primary_vals = [max(n, c) for n, c in zip(sasd_n_vals, sasd_c_vals)]
    ax.plot(valid_pos, primary_vals, 'ko-', label='Primary Reach (Bottleneck)', linewidth=3, zorder=3)
    
    for i, (c_pos, strand) in enumerate(cytosines_nts):
        if c_pos in valid_pos:
            idx = valid_pos.index(c_pos)
            is_target = (args.target_pos is not None and c_pos == args.target_pos)
            color = 'magenta' if is_target else 'green'
            edge = 'black' if is_target else 'darkgreen'
            label = 'Target Cytosine (NTS)' if is_target else 'NTS Cytosine'
            
            # Helper to deduplicate label
            lbl = None
            if label not in [l.get_label() for l in ax.get_lines()] and label not in [l.get_label() for l in ax.collections]:
                 lbl = label
            
            ax.plot(c_pos, primary_vals[idx], 'o' if not is_target else '*', color=color, markersize=12 if is_target else 6, 
                    markeredgecolor=edge, label=lbl, zorder=6 if is_target else 5)

    for c_pos, strand in cytosines_ts:
        if c_pos in valid_pos:
            idx = valid_pos.index(c_pos)
            is_target = (args.target_pos is not None and c_pos == args.target_pos)
            if is_target:
                ax.plot(c_pos, primary_vals[idx], '*', color='magenta', markersize=12, 
                        markeredgecolor='black', label='Target Cytosine (TS)', zorder=6)
            else:
                ax.plot(c_pos, primary_vals[idx], 'o', color='lightgreen', markersize=6, 
                        markeredgecolor='darkgreen', zorder=5)

    ax.set_ylabel('SASD Distance (Å)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_title(f"MitoReach: Structural Accessibility (Linker: {args.linker_type.upper()})", fontsize=14, fontweight='bold')
    
    ax.set_xlabel('Spacer Position', fontsize=12, fontweight='bold')
    x_labels = [f"{r[0]}\n{r[1]}/{r[2]}" for r in valid_results]
    ax.set_xticks(valid_pos)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_xlim(0.5, total_len + 0.5)
    
    plt.tight_layout()
    
    # Save graph in requested format(s)
    base_output = os.path.splitext(args.output)[0]
    formats_to_save = []
    
    if args.graph_format == 'all':
        formats_to_save = ['png', 'svg', 'pdf']
    else:
        formats_to_save = [args.graph_format]
    
    for fmt in formats_to_save:
        output_file = f"{base_output}.{fmt}"
        if fmt == 'png':
            plt.savefig(output_file, dpi=150, format='png')
        elif fmt == 'svg':
            plt.savefig(output_file, format='svg')
        elif fmt == 'pdf':
            plt.savefig(output_file, format='pdf')
        print(f"✅ Saved {fmt.upper()} plot to {output_file}")

    # Recommendation Logic
    if args.target_pos is not None:
        if args.target_pos not in valid_pos:
            print(f"\n[!] Target position {args.target_pos} is invalid (outside spacer).")
        else:
            idx = valid_pos.index(args.target_pos)
            s_n = score_n_vals[idx]
            s_t = score_t_vals[idx]
            
            # Determine strand of target
            base_n = valid_results[idx][1]
            base_t = valid_results[idx][2]
            
            target_strand = "Unknown"
            if base_n == 'C': target_strand = "NTS"
            if base_t == 'C': target_strand = "TS"
            if base_n == 'C' and base_t == 'C': target_strand = "Both"
            
            print(f"\n[Target Inference] Position {args.target_pos} Bases: {base_n}/{base_t} -> Target Strand: {target_strand}")
            
            final_score = max(s_n, s_t)
            is_editable = final_score > 0
            
            if not is_editable:
                print(f"[Recommendation] Target at Pos {args.target_pos} is NOT editable or not a C.")
            else:
                score_str = f"NTS={int(s_n)}" if target_strand=="NTS" else f"TS={int(s_t)}"
                if target_strand=="Both": score_str = f"NTS={int(s_n)}, TS={int(s_t)}"
                print(f"[Success] Target at Pos {args.target_pos} IS editable ({score_str}).")

    # Print Tabular Summary
    print("\n### MitoReach: Structural Reach Simulation Results")
    print(f"| Position | Base (NTS/TS) | NTS Score | TS Score | Bio Window | Status |")
    print("| :---: | :---: | :---: | :---: | :---: | :--- |")
    for i, pos in enumerate(valid_pos):
        sn = score_n_vals[i]
        st = score_t_vals[i]
        base_n = valid_results[i][1]
        base_t = valid_results[i][2]
        dn = sasd_n_vals[i]
        dc = sasd_c_vals[i]
        max_d = max(dn, dc)
        
        bio_window = "-"
        if 8 <= pos <= 13 and 3 <= pos <= 8: bio_window = "Both"
        elif 8 <= pos <= 13: bio_window = "NTS (8-13)"
        elif 3 <= pos <= 8: bio_window = "TS (3-8)"
        
        status = []
        if sn > 0: status.append(f"NTS:✅({int(sn)})")
        if st > 0: status.append(f"TS:✅({int(st)})")
        if not status: 
            if max_d <= reach_threshold:
                 status.append("❌ Wrong Window")
            elif max_d > reach_threshold: 
                 status.append("❌ Out of Reach")
            elif base_n!='C' and base_t!='C': status.append("No C")
            else: status.append("❌ Low Score")
            
        print(f"| {pos} | {base_n}/{base_t} | {int(sn)} | {int(st)} | {bio_window} | {', '.join(status)} |")
    
    # Export to CSV if requested
    if args.export_csv:
        csv_file = f"{base_output}_data.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Position', 'Base_NTS', 'Base_TS', 'Context_NTS', 'Context_TS',
                'SASD_N (Å)', 'SASD_C (Å)', 'Primary_SASD (Å)',
                'NTS_Score', 'TS_Score', 'Bio_Window', 'Status'
            ])
            
            for i, pos in enumerate(valid_pos):
                sn = score_n_vals[i]
                st = score_t_vals[i]
                base_n = valid_results[i][1]
                base_t = valid_results[i][2]
                ctx_n = valid_results[i][7] or '-'
                ctx_t = valid_results[i][8] or '-'
                dn = sasd_n_vals[i]
                dc = sasd_c_vals[i]
                max_d = max(dn, dc)
                
                bio_window = "-"
                if 8 <= pos <= 13 and 3 <= pos <= 8: bio_window = "Both"
                elif 8 <= pos <= 13: bio_window = "NTS (8-13)"
                elif 3 <= pos <= 8: bio_window = "TS (3-8)"
                
                status = []
                if sn > 0: status.append(f"NTS:✅({int(sn)})")
                if st > 0: status.append(f"TS:✅({int(st)})")
                if not status:
                    if max_d <= reach_threshold: status.append("Wrong Window")
                    elif max_d > reach_threshold: status.append("Out of Reach")
                    elif base_n!='C' and base_t!='C': status.append("No C")
                    else: status.append("Low Score")
                
                writer.writerow([
                    pos, base_n, base_t, ctx_n, ctx_t,
                    f"{dn:.2f}", f"{dc:.2f}", f"{max_d:.2f}",
                    int(sn), int(st), bio_window, ', '.join(status)
                ])
        
        print(f"\n✅ Exported CSV data to {csv_file}")
    
    # Export to JSON if requested
    if args.export_json:
        json_file = f"{base_output}_analysis.json"
        
        json_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "mitoreach_version": "1.0",
                "linker_type": args.linker_type,
                "reach_threshold": reach_threshold,
                "optimal_distance": optimal_dist,
                "spacer_sequence_nts": spacer_seq_nts,
                "spacer_sequence_ts": spacer_seq_ts,
                "spacer_length": len(spacer_seq_nts),
                "target_position": args.target_pos
            },
            "positions": [],
            "summary": {
                "total_positions": len(valid_pos),
                "editable_positions_nts": sum(1 for s in score_n_vals if s > 0),
                "editable_positions_ts": sum(1 for s in score_t_vals if s > 0),
                "cytosines_nts": [(p, s) for p, s in cytosines_nts],
                "cytosines_ts": [(p, s) for p, s in cytosines_ts]
            }
        }
        
        for i, pos in enumerate(valid_pos):
            sn = score_n_vals[i]
            st = score_t_vals[i]
            base_n = valid_results[i][1]
            base_t = valid_results[i][2]
            ctx_n = valid_results[i][7]
            ctx_t = valid_results[i][8]
            dn = sasd_n_vals[i]
            dc = sasd_c_vals[i]
            max_d = max(dn, dc)
            
            bio_window = None
            if 8 <= pos <= 13 and 3 <= pos <= 8: bio_window = "Both"
            elif 8 <= pos <= 13: bio_window = "NTS"
            elif 3 <= pos <= 8: bio_window = "TS"
            
            is_editable_nts = sn > 0
            is_editable_ts = st > 0
            
            position_data = {
                "position": pos,
                "bases": {"nts": base_n, "ts": base_t},
                "context": {"nts": ctx_n, "ts": ctx_t},
                "sasd": {
                    "distance_n": round(dn, 2),
                    "distance_c": round(dc, 2),
                    "primary": round(max_d, 2)
                },
                "scores": {
                    "nts": int(sn),
                    "ts": int(st)
                },
                "editability": {
                    "nts": is_editable_nts,
                    "ts": is_editable_ts,
                    "any_strand": is_editable_nts or is_editable_ts
                },
                "bio_window": bio_window,
                "is_target": (args.target_pos == pos) if args.target_pos else False
            }
            
            json_data["positions"].append(position_data)
        
        # Add target-specific recommendation if applicable
        if args.target_pos and args.target_pos in valid_pos:
            idx = valid_pos.index(args.target_pos)
            json_data["target_recommendation"] = {
                "position": args.target_pos,
                "is_editable": (score_n_vals[idx] > 0) or (score_t_vals[idx] > 0),
                "best_score": max(score_n_vals[idx], score_t_vals[idx]),
                "recommended_architecture": "LNRC" if score_n_vals[idx] > score_t_vals[idx] else "LNRN"
            }
        
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"✅ Exported JSON analysis to {json_file}")


if __name__ == "__main__":
    main()

