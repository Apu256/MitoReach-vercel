#!/usr/bin/env python3
"""
TC Motif Scanner for MitoReach Analysis
Identifies all 5'-TC-3' motifs on both strands within a sequence region.
"""

from typing import List, Dict, Tuple, Optional
import re


class TCMotifScanner:
    """Scanner for identifying TC motifs in DNA sequences."""
    
    def __init__(self):
        """Initialize TC motif scanner."""
        self.complement_map = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    
    def get_complement(self, sequence: str) -> str:
        """
        Get complement of DNA sequence.
        
        Args:
            sequence: DNA sequence
        
        Returns:
            Complement sequence
        """
        return ''.join([self.complement_map.get(base, 'N') for base in sequence])
    
    def get_reverse_complement(self, sequence: str) -> str:
        """
        Get reverse complement of DNA sequence.
        
        Args:
            sequence: DNA sequence
        
        Returns:
            Reverse complement sequence
        """
        return self.get_complement(sequence)[::-1]
    
    def find_tc_motifs(self, sequence: str, start_position: int = 1) -> Dict[str, List[Dict]]:
        """
        Find all TC motifs on both strands.
        
        Args:
            sequence: DNA sequence to scan
            start_position: Genomic position of first base (1-indexed)
        
        Returns:
            Dictionary with 'NTS' and 'TS' keys containing lists of motif info
        """
        sequence = sequence.upper()
        complement_seq = self.get_complement(sequence)
        
        results = {
            'NTS': [],  # Non-template strand (forward)
            'TS': []    # Template strand (reverse complement)
        }
        
        # Scan NTS (forward strand) for TC motifs
        for match in re.finditer(r'TC', sequence):
            pos_in_seq = match.start()  # 0-indexed position in sequence
            genomic_pos = start_position + pos_in_seq  # Genomic position
            cytosine_pos = genomic_pos + 1  # Position of C in TC
            
            # Get context (±5 bp around TC)
            context_start = max(0, pos_in_seq - 5)
            context_end = min(len(sequence), pos_in_seq + 7)
            context = sequence[context_start:context_end]
            
            results['NTS'].append({
                'motif': 'TC',
                'strand': 'NTS',
                'position': cytosine_pos,  # Position of target C
                'tc_start': genomic_pos,    # Position of T
                'context': context,
                'context_start': start_position + context_start
            })
        
        # Scan TS (complement strand) for TC motifs
        # On complement, TC motif (5'-TC-3') means there's a 5'-GA-3' on the forward strand
        # due to antiparallel nature: 
        # Forward: 5'-G(i) A(i+1)-3'
        # Reverse: 3'-C(i) T(i+1)-5' -> 5'-T(i+1) C(i)-3'
        for match in re.finditer(r'GA', sequence):
            pos_in_seq = match.start()  # 0-indexed position of G in GA
            genomic_pos_g = start_position + pos_in_seq
            genomic_pos_a = genomic_pos_g + 1
            
            # For the TS motif 5'-TC-3':
            # T is at the 'A' position (genomic_pos_a)
            # C is at the 'G' position (genomic_pos_g)
            cytosine_pos = genomic_pos_g 
            tc_start = genomic_pos_a # T is at i+1
            
            # Get context on complement strand (must also be reversed for 5'-3' view)
            # But for simplicity in this tool, we extract around the genomic position
            context_start = max(0, pos_in_seq - 5)
            context_end = min(len(sequence), pos_in_seq + 7)
            context_fwd = sequence[context_start:context_end]
            context_rev = self.get_reverse_complement(context_fwd)
            
            results['TS'].append({
                'motif': 'TC',
                'strand': 'TS',
                'position': cytosine_pos,  # Position of target C on TS
                'tc_start': tc_start,     # Position of T on TS
                'context': context_rev,
                'context_start': genomic_pos_g # Approximate
            })
        
        return results
    
    def check_position_has_tc(self, sequence: str, start_position: int, target_position: int) -> Tuple[bool, Optional[Dict]]:
        """
        Check if a specific genomic position has a TC motif.
        
        Args:
            sequence: DNA sequence
            start_position: Genomic position of first base (1-indexed)
            target_position: Target genomic position to check
        
        Returns:
            Tuple of (has_tc, motif_info)
            has_tc: True if target position has TC
            motif_info: Dictionary with motif details or None
        """
        # Find all TC motifs
        all_motifs = self.find_tc_motifs(sequence, start_position)
        
        # Check NTS
        for motif in all_motifs['NTS']:
            if motif['position'] == target_position:
                return True, motif
        
        # Check TS
        for motif in all_motifs['TS']:
            if motif['position'] == target_position:
                return True, motif
        
        return False, None
    
    def find_nearest_tc_motifs(self, all_motifs: Dict[str, List[Dict]], 
                               target_position: int, max_distance: int = 50) -> List[Dict]:
        """
        Find TC motifs nearest to target position.
        
        Args:
            all_motifs: Dictionary of TC motifs from find_tc_motifs
            target_position: Target genomic position
            max_distance: Maximum distance to consider (default 50 bp)
        
        Returns:
            List of nearest TC motifs sorted by distance
        """
        nearby_motifs = []
        
        # Collect all motifs with distance
        for strand in ['NTS', 'TS']:
            for motif in all_motifs[strand]:
                distance = abs(motif['position'] - target_position)
                if distance <= max_distance and distance > 0:  # Exclude exact match
                    motif_with_dist = motif.copy()
                    motif_with_dist['distance'] = distance
                    motif_with_dist['direction'] = 'downstream' if motif['position'] > target_position else 'upstream'
                    nearby_motifs.append(motif_with_dist)
        
        # Sort by distance
        nearby_motifs.sort(key=lambda x: x['distance'])
        
        return nearby_motifs
    
    def summarize_motifs(self, all_motifs: Dict[str, List[Dict]]) -> str:
        """
        Create summary string of TC motifs found.
        
        Args:
            all_motifs: Dictionary of TC motifs from find_tc_motifs
        
        Returns:
            Summary string
        """
        nts_count = len(all_motifs['NTS'])
        ts_count = len(all_motifs['TS'])
        total = nts_count + ts_count
        
        summary = f"Found {total} TC motifs: {nts_count} on NTS, {ts_count} on TS\n"
        
        if nts_count > 0:
            summary += f"  NTS positions: {', '.join([str(m['position']) for m in all_motifs['NTS']])}\n"
        if ts_count > 0:
            summary += f"  TS positions:  {', '.join([str(m['position']) for m in all_motifs['TS']])}\n"
        
        return summary


def main():
    """Test the TC motif scanner."""
    import sys
    
    scanner = TCMotifScanner()
    
    # Test with a sample sequence
    test_seq = "ATCGATCGAAGTCGATCGATCGATC"
    print(f"Test sequence: {test_seq}")
    print(f"Complement:    {scanner.get_complement(test_seq)}")
    
    motifs = scanner.find_tc_motifs(test_seq, start_position=1000)
    print(f"\n{scanner.summarize_motifs(motifs)}")
    
    # Check specific position
    target_pos = 1002
    has_tc, motif_info = scanner.check_position_has_tc(test_seq, 1000, target_pos)
    print(f"\nPosition {target_pos} has TC: {has_tc}")
    if has_tc:
        print(f"  Strand: {motif_info['strand']}")
        print(f"  Context: {motif_info['context']}")
    
    # Find nearest
    if not has_tc:
        nearest = scanner.find_nearest_tc_motifs(motifs, target_pos, max_distance=50)
        if nearest:
            print(f"\nNearest TC motifs:")
            for m in nearest[:3]:
                print(f"  Position {m['position']} ({m['strand']}) - {m['distance']} bp {m['direction']}")


if __name__ == '__main__':
    main()
