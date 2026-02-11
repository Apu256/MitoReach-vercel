#!/usr/bin/env python3
"""
GenBank Genome Parser for mtDNA MitoReach Analysis
Parses GenBank files, extracts gene coordinates, and handles circular chromosomes.
"""

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature
from typing import Dict, List, Tuple, Optional
import re


class GenomeParser:
    """Parser for mitochondrial genome GenBank files."""
    
    def __init__(self, genbank_file: str):
        """
        Initialize parser with GenBank file.
        
        Args:
            genbank_file: Path to GenBank file
        """
        self.genbank_file = genbank_file
        self.record: Optional[SeqRecord] = None
        self.genome_length: int = 0
        self.genes: Dict[str, Dict] = {}
        
        self._load_genbank()
    
    def _load_genbank(self):
        """Load and parse GenBank file."""
        try:
            self.record = SeqIO.read(self.genbank_file, "genbank")
            self.genome_length = len(self.record.seq)
            self._extract_genes()
            print(f"Loaded genome: {self.record.description}")
            print(f"Length: {self.genome_length} bp")
            print(f"Found {len(self.genes)} genes")
        except Exception as e:
            raise ValueError(f"Failed to load GenBank file '{self.genbank_file}': {e}")
    
    def _extract_genes(self):
        """Extract all gene features from GenBank record."""
        for feature in self.record.features:
            # Look for gene or CDS features
            if feature.type in ['gene', 'CDS']:
                gene_name = None
                
                # Try different qualifiers for gene name
                for qualifier in ['gene', 'gene_synonym', 'label', 'locus_tag']:
                    if qualifier in feature.qualifiers:
                        gene_name = feature.qualifiers[qualifier][0]
                        break
                
                if gene_name:
                    # Clean up gene name (remove spaces, make uppercase)
                    gene_name = gene_name.strip().upper()
                    
                    # Store gene info
                    if gene_name not in self.genes:
                        self.genes[gene_name] = {
                            'name': gene_name,
                            'start': int(feature.location.start) + 1,  # 1-indexed
                            'end': int(feature.location.end),
                            'strand': feature.location.strand,
                            'type': feature.type,
                            'product': feature.qualifiers.get('product', [''])[0]
                        }
    
    def list_genes(self) -> List[Dict]:
        """
        List all genes in genome.
        
        Returns:
            List of gene dictionaries with name, coordinates, strand
        """
        return sorted(self.genes.values(), key=lambda x: x['start'])
    
    def get_gene_info(self, gene_name: str) -> Optional[Dict]:
        """
        Get information for a specific gene.
        
        Args:
            gene_name: Gene name (case-insensitive)
        
        Returns:
            Dictionary with gene info or None if not found
        """
        gene_name = gene_name.strip().upper()
        return self.genes.get(gene_name)
    
    def extract_sequence(self, start: int, end: int, circular: bool = True) -> str:
        """
        Extract sequence from genome with circular chromosome support.
        
        Args:
            start: Start position (1-indexed, inclusive)
            end: End position (1-indexed, inclusive)
            circular: Handle wrapping for circular chromosomes
        
        Returns:
            Extracted sequence as string
        """
        # Convert to 0-indexed for BioPython
        start_0 = start - 1
        end_0 = end
        
        # Handle circular wrapping
        if circular and end > self.genome_length:
            # Wraps around: extract end portion + beginning portion
            seq1 = str(self.record.seq[start_0:])
            seq2 = str(self.record.seq[:end_0 - self.genome_length])
            return seq1 + seq2
        elif circular and start < 1:
            # Wraps at beginning
            seq1 = str(self.record.seq[self.genome_length + start_0:])
            seq2 = str(self.record.seq[:end_0])
            return seq1 + seq2
        else:
            # Normal extraction
            return str(self.record.seq[start_0:end_0])
    
    def get_scanning_window(self, position: int, window_size: int = 50) -> Tuple[int, int, str]:
        """
        Get scanning window around target position.
        
        Args:
            position: Target position (1-indexed)
            window_size: Window size in bp (±window_size)
        
        Returns:
            Tuple of (start, end, sequence)
        """
        start = position - window_size
        end = position + window_size
        
        # Get sequence (handles circular wrapping)
        sequence = self.extract_sequence(start, end, circular=True)
        
        return start, end, sequence
    
    def validate_position(self, position: int, gene_name: Optional[str] = None) -> Dict:
        """
        Validate a genomic position and check if it's within a gene.
        
        Args:
            position: Genomic position (1-indexed)
            gene_name: Optional gene name to validate against
        
        Returns:
            Dictionary with validation results
        """
        result = {
            'valid': False,
            'position': position,
            'in_gene': False,
            'gene_info': None,
            'message': ''
        }
        
        # Check if position is within genome
        if position < 1 or position > self.genome_length:
            result['message'] = f"Position {position} is out of range (1-{self.genome_length})"
            return result
        
        result['valid'] = True
        
        # If gene name provided, check if position is within that gene
        if gene_name:
            gene_info = self.get_gene_info(gene_name)
            if not gene_info:
                result['message'] = f"Gene '{gene_name}' not found in genome"
                return result
            
            # Check if position is within gene boundaries
            if gene_info['start'] <= position <= gene_info['end']:
                result['in_gene'] = True
                result['gene_info'] = gene_info
                offset = position - gene_info['start'] + 1
                result['message'] = (f"Position {position} is within {gene_name} "
                                   f"({gene_info['start']}-{gene_info['end']}, offset +{offset} bp)")
            else:
                result['message'] = (f"Position {position} is NOT within {gene_name} "
                                   f"({gene_info['start']}-{gene_info['end']})")
        else:
            # Find which gene(s) contain this position
            containing_genes = []
            for gene in self.genes.values():
                if gene['start'] <= position <= gene['end']:
                    containing_genes.append(gene['name'])
            
            if containing_genes:
                result['in_gene'] = True
                result['message'] = f"Position {position} is within: {', '.join(containing_genes)}"
            else:
                result['message'] = f"Position {position} is in intergenic region"
        
        return result


def main():
    """Test the genome parser."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 genome_parser.py <genbank_file> [--list-genes] [--gene <name>] [--position <pos>]")
        sys.exit(1)
    
    parser = GenomeParser(sys.argv[1])
    
    if '--list-genes' in sys.argv:
        print("\n=== Genes in Genome ===")
        for gene in parser.list_genes():
            print(f"{gene['name']:15s} {gene['start']:6d}-{gene['end']:6d}  "
                  f"({gene['end']-gene['start']+1:4d} bp)  {gene['product']}")
    
    if '--gene' in sys.argv:
        idx = sys.argv.index('--gene')
        gene_name = sys.argv[idx + 1]
        gene_info = parser.get_gene_info(gene_name)
        if gene_info:
            print(f"\n=== Gene: {gene_name} ===")
            print(f"Coordinates: {gene_info['start']}-{gene_info['end']}")
            print(f"Length: {gene_info['end'] - gene_info['start'] + 1} bp")
            print(f"Strand: {'+' if gene_info['strand'] == 1 else '-'}")
            print(f"Product: {gene_info['product']}")
        else:
            print(f"Gene '{gene_name}' not found")
    
    if '--position' in sys.argv:
        idx = sys.argv.index('--position')
        position = int(sys.argv[idx + 1])
        
        gene_name = None
        if '--gene' in sys.argv:
            idx_gene = sys.argv.index('--gene')
            gene_name = sys.argv[idx_gene + 1]
        
        validation = parser.validate_position(position, gene_name)
        print(f"\n=== Position Validation ===")
        print(validation['message'])
        
        # Show scanning window
        start, end, seq = parser.get_scanning_window(position, 50)
        print(f"\nScanning window (±50bp): [{start}, {end}]")
        print(f"Sequence length: {len(seq)} bp")
        print(f"First 50 bp: {seq[:50]}")


if __name__ == '__main__':
    main()
