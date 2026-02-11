import sys
import math
import heapq
import numpy as np

# Constants
GRID_RESOLUTION = 1.5  # Angstroms (Trade-off: 1.0 is better but slower, 2.0 is fast but coarse)
VDW_RADIUS_DEFAULT = 1.8
PROBE_RADIUS = 2.0 # Distance from protein surface to consider "surface layer"
# Standard VdW radii (approximate)
ATOM_RADII = {
    'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80, 'H': 1.20, 'P': 1.80
}

def get_atom_radius(atom_name):
    element = atom_name[0:1] # Simple element extraction
    return ATOM_RADII.get(element, VDW_RADIUS_DEFAULT)

class SASDCalculator:
    def __init__(self, pdb_file, resolution=GRID_RESOLUTION):
        self.pdb_file = pdb_file
        self.resolution = resolution
        self.atoms = []
        self.min_coords = None
        self.max_coords = None
        self.grid = None
        self.grid_shape = None
        self.origin = None
        
    def parse_pdb(self):
        print("Parsing PDB (All Atoms)...")
        atoms = []
        
        with open(self.pdb_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    atom_name = line[12:16].strip()
                    chain = line[21]
                    res_id = int(line[22:26])
                    
                    atoms.append({
                        'coord': np.array([x, y, z]),
                        'radius': get_atom_radius(atom_name),
                        'chain': chain,
                        'res_id': res_id,
                        'name': atom_name
                    })
        self.atoms = atoms
        coords = np.array([a['coord'] for a in atoms])
        self.min_coords = coords.min(axis=0) - 10.0 # Buffer
        self.max_coords = coords.max(axis=0) + 10.0
        self.origin = self.min_coords
        print(f"Parsed {len(self.atoms)} atoms.")
        
    def build_grid(self):
        print("Building Grid...")
        # Calculate grid dimensions
        dims = (self.max_coords - self.min_coords) / self.resolution
        self.grid_shape = np.ceil(dims).astype(int)
        
        # 0: Solvent, 1: Protein
        self.grid = np.zeros(self.grid_shape, dtype=np.int8)
        
        print(f"Grid Shape: {self.grid_shape}")
        
        # Mark protein voxels
        # Optimization: Iterate over atoms and mark bounding box of each atom
        for atom in self.atoms:
            center_grid = (atom['coord'] - self.origin) / self.resolution
            radius_grid = atom['radius'] / self.resolution
            
            # Define bounding box for this atom in grid coords
            min_idx = np.floor(center_grid - radius_grid).astype(int)
            max_idx = np.ceil(center_grid + radius_grid).astype(int)
            
            # Clip to grid bounds
            min_idx = np.maximum(min_idx, 0)
            max_idx = np.minimum(max_idx, self.grid_shape - 1)
            
            # Iterate over the bounding box
            # This is the slow part in pure Python, but numpy can help
            # Create a meshgrid for the box
            ranges = [np.arange(min_idx[i], max_idx[i]+1) for i in range(3)]
            if any(len(r) == 0 for r in ranges): continue
            
            # We can just iterate simply for now or use vectorization
            # Vectorized approach for the box:
            X, Y, Z = np.meshgrid(*ranges, indexing='ij')
            grid_coords = np.stack([X, Y, Z], axis=-1)
            
            # Convert back to world coords relative to atom center
            # Distance check
            # (idx * res + origin - atom_coord)^2 < r^2
            # Let's do it in grid space
            dist_sq = np.sum((grid_coords - center_grid)**2, axis=-1)
            mask = dist_sq <= radius_grid**2
            
            # Update grid
            self.grid[X[mask], Y[mask], Z[mask]] = 1
            
        print("Grid built. Protein voxels marked.")

    def get_surface_voxels(self):
        print("Identifying Surface Voxels...")
        # Surface voxels are Solvent (0) voxels that are within PROBE_RADIUS of Protein (1)
        # Alternatively, simple adjacency: Solvent voxels with at least one Protein neighbor
        
        # Let's use distance transform or simply dilation if we had scipy.ndimage
        # Without scipy, we can iterate or use a convolution-like approach.
        
        # Approach:
        # 1. Find all Protein voxels.
        # 2. Check their neighbors. If neighbor is 0, it's a surface candidate.
        # 3. To allow movement *around* the surface, we need a layer of solvent.
        
        # Let's define "Accessible Space" as any voxel that is 0.
        # But we want the path to "hug" the surface.
        # So we define the graph nodes as: Any Solvent voxel within X distance of Protein.
        
        # Optimization:
        # Iterate over all Protein voxels (1).
        # Mark neighbors within PROBE_RADIUS as "Surface" (2).
        # But don't overwrite Protein (1).
        
        protein_indices = np.argwhere(self.grid == 1)
        surface_grid = np.zeros_like(self.grid)
        
        # Radius in grid units
        probe_res = PROBE_RADIUS / self.resolution
        search_rad = int(np.ceil(probe_res))
        
        # This might be slow if we iterate every protein voxel.
        # Instead, let's just find the boundary.
        # A voxel is surface if it is 0 and has a neighbor that is 1.
        
        # Let's stick to the "Surface Layer" concept.
        # We want to find the shortest path in the set of voxels V_surf where:
        # V_surf = {v | grid[v] == 0 AND dist(v, protein) < threshold}
        
        # To do this efficiently without scipy.ndimage.distance_transform_edt:
        # We can collect all 0-neighbors of 1-voxels.
        
        # Shift arrays to find boundary
        # Neighbors: 6-connectivity
        is_protein = (self.grid == 1)
        is_surface = np.zeros_like(is_protein)
        
        shifts = [
            (1,0,0), (-1,0,0),
            (0,1,0), (0,-1,0),
            (0,0,1), (0,0,-1)
        ]
        
        for dx, dy, dz in shifts:
            # Shift the protein mask
            shifted = np.roll(is_protein, shift=(dx, dy, dz), axis=(0, 1, 2))
            # Handle boundaries (roll wraps around, so we should mask edges, but with buffer it's fine)
            
            # If a voxel is NOT protein, but its neighbor IS protein, it's surface
            is_surface |= (shifted & (~is_protein))
            
        # This gives us the immediate shell (1-voxel thick).
        # For pathfinding, 1-voxel thick might be disconnected diagonally.
        # Let's dilate this surface layer by 1 more voxel to ensure connectivity and "Probe Radius"
        
        surface_layer_1 = is_surface.copy()
        for dx, dy, dz in shifts:
            shifted = np.roll(surface_layer_1, shift=(dx, dy, dz), axis=(0, 1, 2))
            is_surface |= (shifted & (~is_protein))
            
        self.surface_indices = np.argwhere(is_surface)
        # Create a set for fast lookup
        self.surface_set = set(map(tuple, self.surface_indices))
        
        print(f"Identified {len(self.surface_indices)} surface voxels.")
        return self.surface_set

    def coord_to_grid(self, coord):
        return tuple(np.round((coord - self.origin) / self.resolution).astype(int))

    def grid_to_coord(self, grid_idx):
        return self.origin + np.array(grid_idx) * self.resolution

    def find_nearest_surface_voxel(self, target_coord):
        # Convert target to grid
        target_idx = self.coord_to_grid(target_coord)
        
        # BFS to find nearest voxel in self.surface_set
        # Or just brute force if close
        
        # Simple BFS
        queue = [(target_idx, 0)]
        visited = {target_idx}
        
        # Limit search depth
        max_depth = 20 
        
        best_voxel = None
        min_dist = float('inf')
        
        # If target is already in surface set
        if target_idx in self.surface_set:
            return target_idx
            
        # BFS
        head = 0
        while head < len(queue):
            curr, depth = queue[head]
            head += 1
            
            if curr in self.surface_set:
                return curr
            
            if depth >= max_depth:
                continue
                
            x, y, z = curr
            neighbors = [
                (x+1, y, z), (x-1, y, z),
                (x, y+1, z), (x, y-1, z),
                (x, y, z+1), (x, y, z-1)
            ]
            
            for n in neighbors:
                if n not in visited:
                    # Check bounds
                    if 0 <= n[0] < self.grid_shape[0] and \
                       0 <= n[1] < self.grid_shape[1] and \
                       0 <= n[2] < self.grid_shape[2]:
                        visited.add(n)
                        queue.append((n, depth + 1))
                        
        return None

    def calculate_sasd(self, start_res, end_res):
        # start_res: (chain, res_id)
        # Find CB atoms, fallback to CA
        def get_target_atom(chain, res_id):
            # Try CB first
            atom = next((a for a in self.atoms if a['chain'] == chain and a['res_id'] == res_id and a['name'] == 'CB'), None)
            if not atom:
                # Fallback to CA
                atom = next((a for a in self.atoms if a['chain'] == chain and a['res_id'] == res_id and a['name'] == 'CA'), None)
            return atom

        start_atom = get_target_atom(start_res[0], start_res[1])
        end_atom = get_target_atom(end_res[0], end_res[1])
        
        if not start_atom or not end_atom:
            print("Start or End atom not found.")
            return None
            
        print(f"Start Atom: {start_atom['name']} at {start_atom['coord']}")
        print(f"End Atom: {end_atom['name']} at {end_atom['coord']}")
        
        start_voxel = self.find_nearest_surface_voxel(start_atom['coord'])
        end_voxel = self.find_nearest_surface_voxel(end_atom['coord'])
        
        if not start_voxel or not end_voxel:
            print("Could not map atoms to surface voxels.")
            return None
            
        print(f"Start Voxel: {start_voxel}")
        print(f"End Voxel: {end_voxel}")
        
        # Dijkstra
        print("Running Dijkstra...")
        pq = [(0, start_voxel)]
        distances = {start_voxel: 0}
        
        # Pre-compute neighbors offsets and weights
        # 26-connectivity
        offsets = []
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                for z in [-1, 0, 1]:
                    if x==0 and y==0 and z==0: continue
                    dist = math.sqrt(x*x + y*y + z*z) * self.resolution
                    offsets.append(((x, y, z), dist))
                    
        while pq:
            d, curr = heapq.heappop(pq)
            
            if curr == end_voxel:
                print("Path found!")
                return d
            
            if d > distances.get(curr, float('inf')):
                continue
            
            cx, cy, cz = curr
            
            for (dx, dy, dz), weight in offsets:
                neighbor = (cx + dx, cy + dy, cz + dz)
                
                if neighbor in self.surface_set:
                    new_dist = d + weight
                    if new_dist < distances.get(neighbor, float('inf')):
                        distances[neighbor] = new_dist
                        heapq.heappush(pq, (new_dist, neighbor))
                        
        print("No path found.")
        return None

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Calculate SASD between two residues.')
    parser.add_argument('pdb_file', help='Path to PDB file')
    parser.add_argument('--start_chain', default='A', help='Chain ID for start residue')
    parser.add_argument('--start_res', type=int, default=641, help='Residue ID for start residue')
    parser.add_argument('--end_chain', default='D', help='Chain ID for end residue')
    parser.add_argument('--end_res', type=int, default=1290, help='Residue ID for end residue')
    
    args = parser.parse_args()

    calculator = SASDCalculator(args.pdb_file)
    calculator.parse_pdb()
    calculator.build_grid()
    calculator.get_surface_voxels()
    
    print(f"Calculating SASD from {args.start_chain}:{args.start_res} to {args.end_chain}:{args.end_res}")
    sasd = calculator.calculate_sasd((args.start_chain, args.start_res), (args.end_chain, args.end_res))
    
    if sasd:
        print(f"Calculated SASD: {sasd:.2f} Angstroms")
    else:
        print("Failed to calculate SASD.")

if __name__ == "__main__":
    main()
