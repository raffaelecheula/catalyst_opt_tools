# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from ase import Atoms
from itertools import combinations
from ase.neighborlist import natural_cutoffs
from ase.build.tools import rotation_matrix

from ase_ml_models.utilities import (
    get_connectivity,
    get_edges_list_from_connectivity,
)

# -------------------------------------------------------------------------------------
# ENLARGE SURFACE
# -------------------------------------------------------------------------------------

def enlarge_surface(
    atoms: Atoms,
) -> Atoms:
    """
    Enlarge the structure by adding atoms at the cell boundaries.
    """
    atoms_enlarged = atoms.copy()
    for ii in [-1, 0, 1]:
        for jj in [-1, 0, 1]:
            if ii == jj == 0:
                continue
            atoms_copy = atoms.copy()
            atoms_copy.translate(np.dot([ii, jj, 0], atoms.cell))
            atoms_enlarged += atoms_copy
    atoms_enlarged.pbc = False
    atoms_enlarged.info["indices_original"] = list(range(len(atoms))) * 9
    return atoms_enlarged

# -------------------------------------------------------------------------------------
# GET INDICES FROM BOND CUTOFF
# -------------------------------------------------------------------------------------

def get_indices_from_bond_cutoff(
    atoms: Atoms,
    connectivity: np.ndarray,
    indices_start: list,
    bond_cutoff: int = 1,
    return_list: bool = False,
):
    """
    Get the indices of atoms within a certain number of bonds.
    """
    indices_all = []
    indices_dict = {}
    indices_dict[0] = indices_start
    indices_all += list(indices_dict[0])
    for ii in range(bond_cutoff):
        indices = [
            int(ii) for ii in np.where(connectivity[indices_dict[ii], :] > 0)[1]
        ]
        indices_dict[ii+1] = list(
            dict.fromkeys([jj for jj in indices if jj not in indices_all])
        )
        indices_all += indices_dict[ii+1]
    if return_list is True:
        return indices_all
    else:
        return indices_dict

# -------------------------------------------------------------------------------------
# GET CLUSTER FROM SURFACE
# -------------------------------------------------------------------------------------

def get_cluster_from_surface(
    atoms: Atoms,
    indices_ads: list = [],
    indices_site: list = [],
    method: str = "ase",
    bond_cutoff: int = 1,
    remove_pbc: bool = True,
    **kwargs,
) -> Atoms:
    """
    Get cluster from surface.
    """
    # Get indices to start calculating the cutoffs.
    indices_start = list(dict.fromkeys(indices_ads+indices_site))
    # Enlarge the surface in x and y directions.
    atoms_enlarged = enlarge_surface(atoms=atoms)
    # Calculate connectivity of enlarged surface.
    connectivity = get_connectivity(
        atoms=atoms_enlarged,
        method=method,
        **kwargs,
    )
    # Get indices of atoms within the bond cutoff from starting atoms.
    indices_list = get_indices_from_bond_cutoff(
        atoms=atoms_enlarged,
        connectivity=connectivity,
        indices_start=indices_start,
        bond_cutoff=bond_cutoff,
        return_list=True,
    )
    # Get the indices of the cluster atoms object.
    indices_list = [
        ii for ii in indices_list
        if atoms_enlarged.info["indices_original"][ii] not in indices_ads
    ] + indices_ads
    # Get the cluster atoms object.
    atoms_new = atoms_enlarged[indices_list]
    atoms_new.info["indices_original"] = [
        atoms_enlarged.info["indices_original"][ii] for ii in indices_list
    ]
    if indices_ads:
        atoms_new.info["indices_ads"] = [
            ii for ii, index in enumerate(atoms_new.info["indices_original"])
            if index in indices_ads
        ]
    if indices_site:
        atoms_new.info["indices_site"] = [
            ii for ii, index in enumerate(atoms_new.info["indices_original"])
            if index in indices_site
        ]
    # Get the connectivity of the cluster.
    atoms_new.info["connectivity"] = get_connectivity(
        atoms=atoms_new,
        method=method,
        **kwargs,
    )
    # Reorder the features.
    if "features" in atoms.info.keys():
        atoms_new.info["features"] = (
            atoms.info["features"][atoms_new.info["indices_original"], :]
        )
    # Remove periodic boundary conditions.
    if remove_pbc is True:
        atoms_new.pbc = False
        atoms_new.cell = None
    # Return the cluster atoms object.
    return atoms_new

# -------------------------------------------------------------------------------------
# GET SURFACE EDGES
# -------------------------------------------------------------------------------------

def get_surface_edges(
    connectivity: list,
    indices_surf: list,
) -> list:
    """
    Get edges of the surface from the connectivity list.
    """
    edges_list = get_edges_list_from_connectivity(connectivity=connectivity)
    return [
        (aa, bb) for (aa, bb) in edges_list if {aa, bb}.issubset(indices_surf)
    ]

# -------------------------------------------------------------------------------------
# GET ADSORPTION SITES
# -------------------------------------------------------------------------------------

def get_adsorption_sites(
    indices_surf: list,
    edges_surf: list,
) -> dict:
    """
    Get adsorption sites for the surface.
    """
    # Prepare sites dictionary.
    sites_dict = {}
    indices_surf = sorted(indices_surf)
    edges_set = set(tuple(sorted(edge)) for edge in edges_surf)
    # Get top sites.
    sites_dict["top"] = [[aa] for aa in indices_surf]
    # Get bridge sites.
    sites_dict["brg"] = [[aa, bb] for (aa, bb) in edges_surf]
    # Get 3-fold hollow sites.
    sites_dict["3fh"] = [
        [aa, bb, cc] for (aa, bb, cc) in combinations(indices_surf, 3)
        if {(aa, bb), (bb, cc), (aa, cc)}.issubset(edges_set)
    ]
    # Get 4-fold hollow sites.
    sites_dict["4fh"] = [
        list(indices) for indices in combinations(indices_surf, 4)
        if sum((aa, bb) in edges_set for aa, bb in combinations(indices, 2)) == 4
        and all(
            sum((aa, bb) in edges_set or (bb, aa) in edges_set for bb in indices) == 2
            for aa in indices
        )
    ]
    # Reorder 4-fold hollow sites to have closed loops.
    sites_dict["4fh"] = [
        [aa, bb, cc, dd] if (bb, cc) in edges_set else [aa, bb, dd, cc]
        for (aa, bb, cc, dd) in sites_dict["4fh"]
    ]
    # Return the sites dictionary.
    return sites_dict

# -------------------------------------------------------------------------------------
# GET ROTATED LISTS
# -------------------------------------------------------------------------------------

def rotated_lists(
    list_of_lists: list,
) -> list:
    """
    Get rotated lists from the input list.
    """
    rotated = [
        [sublist[ii:] + sublist[:ii] for ii in range(len(sublist))]
        for sublist in list_of_lists
    ]
    return [item for sublist in rotated for item in sublist]

# -------------------------------------------------------------------------------------
# GET BIDENTATE SITES
# -------------------------------------------------------------------------------------

def get_bidentate_sites(
    sites_dict: dict,
) -> dict:
    """
    Get bidentate top bridge sites from the adsorption sites.
    """
    sites_bi = {}
    # Get top-top sites.
    sites_bi["top-top"] = [
        [[aa], [bb]] for aa, bb in rotated_lists(sites_dict["brg"])
    ]
    sites_bi["top-top"] += [
        [[aa], [cc]] for aa, bb, cc, dd in rotated_lists(sites_dict["4fh"])
    ]
    # Get top-bridge sites.
    sites_bi["top-brg"] = [
        [[aa], [bb, cc]] for aa, bb, cc in rotated_lists(sites_dict["3fh"])
    ]
    # Get bridge-top sites.
    sites_bi["brg-top"] = [
        [[aa, bb], [cc]] for aa, bb, cc in rotated_lists(sites_dict["3fh"])
    ]
    # Get bridge-bridge sites.
    sites_bi["brg-brg"] = [
        [[aa, bb], [cc, dd]] for aa, bb, cc, dd in rotated_lists(sites_dict["4fh"])
    ]
    return sites_bi

# -------------------------------------------------------------------------------------
# GET PYRAMID HEIGHT 
# -------------------------------------------------------------------------------------

def get_pyramid_height(
    slant: float,
    positions: list,
) -> float:
    """
    Get hight of the n-gon pyramid from the value of the slant and positions
    of the vertices of the n-gon base.
    """
    # Return initial value if only one base point is provided.
    if len(positions) < 2:
        return slant
    # Average length of the base sides of the pyramid.
    baselen = np.mean([
        np.linalg.norm(positions[ii]-positions[ii-1]) for ii in range(len(positions))
    ])
    # Formula for heights of n-gon pyramids.
    d_squared = slant**2 - (baselen/(2*np.sin(np.pi/len(positions))))**2
    return np.sqrt(d_squared) if d_squared > 0.0 else 0.0

# -------------------------------------------------------------------------------------
# GET SITES DIRECTIONS
# -------------------------------------------------------------------------------------

def get_sites_directions(
    atoms_surf: Atoms,
    sites_dict: list,
) -> list:
    """
    Get sites directions.
    """
    directions = {}
    # Get 3fh and 4fh sites.
    for site in [site for name in ["3fh", "4fh"] for site in sites_dict[name]]:
        # Get the positions of the site atoms.
        xx, yy, zz = atoms_surf.positions[site].T
        aa = np.vstack([xx, yy, np.ones_like(xx)]).T
        # Calculate the direction normal to the site plane.
        aa, bb, cc = np.linalg.lstsq(a=aa, b=zz, rcond=None)[0]
        vect = [-aa, -bb, +1]
        # Normalize the direction.
        vect /= np.linalg.norm(vect)
        # Update directions of 3fh and 4fh sites.
        directions[tuple(site)] = directions.get(tuple(site), []) + [vect]
        # Update directions of top sites.
        for top in site:
            directions[top] = directions.get(top, []) + [vect]
        # Update directions of top sites.
        for brg in [sorted((site[ii], site[ii-1])) for ii, _ in enumerate(site)]:
            directions[tuple(brg)] = directions.get(tuple(brg), []) + [vect]
    # Average and normalize the directions for each site.
    for site in directions:
        # Get the average direction for the site.
        directions[site] = np.mean(directions[site], axis=0)
        # Normalize the direction.
        directions[site] /= np.linalg.norm(directions[site])
    # Return the directions dictionary.
    return directions

# -------------------------------------------------------------------------------------
# UPDATE CONNECTIVITY
# -------------------------------------------------------------------------------------

def attach_adsorbate(
    atoms_surf: Atoms,
    atoms_ads: Atoms,
    surf_bound: list,
    site_indices: list,
) -> Atoms:
    """
    Attach adsorbate to the surface and update connectivity.
    """
    # Copy atoms object.
    atoms_surfads = atoms_surf.copy()
    # Add adsorbate to the surface.
    atoms_surfads += atoms_ads
    # Prepare connectivity.
    connectivity = np.zeros((len(atoms_surfads), len(atoms_surfads)), dtype=int)
    # Update with surface connectivity.
    connectivity[:len(atoms_surf), :len(atoms_surf)] = atoms_surf.info["connectivity"]
    # Update with adsorbate connectivity.
    connectivity[len(atoms_surf):, len(atoms_surf):] = atoms_ads.info["connectivity"]
    # Update with the surface-adsorbate connections.
    if len(surf_bound) == 1:
        site_indices = [site_indices]
    for index, indices in zip(surf_bound, site_indices):
        ii = index + len(atoms_surf)
        for jj in indices:
            connectivity[ii, jj] += 1
            connectivity[jj, ii] += 1
    # Update the atoms info with the new connectivity.
    atoms_surfads.info["connectivity"] = connectivity
    # Store the indices of the adsorbate atoms.
    atoms_surfads.info["indices_ads"] = [
        ii + len(atoms_surf) for ii in range(len(atoms_ads))
    ]
    # Return the surface+adsorbate atoms object.
    return atoms_surfads

# -------------------------------------------------------------------------------------
# ADSORPTION MONODENTATE
# -------------------------------------------------------------------------------------

def adsorption_monodentate(
    atoms_mol: Atoms,
    atoms_surf: Atoms,
    surf_bound: list,
    site_indices: list,
) -> Atoms:
    """
    Put mono-dentate adsorbate molecule on one surface site.
    """
    # Copy atoms object.
    atoms_ads = atoms_mol.copy()
    # Get position of the site and bound atom.
    pos_site_atoms = atoms_surf.positions[site_indices]
    pos_site = np.mean(pos_site_atoms, axis=0)
    pos_bound = atoms_ads.positions[surf_bound[0]]
    # Set the position of the bound atom to the origin.
    atoms_ads.translate(-pos_bound)
    # Get distance adsorbate-site.
    slant = natural_cutoffs(atoms_ads)[surf_bound[0]]
    slant += np.mean(natural_cutoffs(atoms_surf[site_indices]))
    # Modify the distance if adsorbate is bound to muptiple atoms.
    distance = get_pyramid_height(slant=slant, positions=pos_site_atoms)
    # Translate adsorbate according to the distance from the site.
    atoms_ads.translate([0., 0., distance])
    # Translate adsorbate to the site position.
    atoms_ads.translate(pos_site)
    # Add adsorbate to the surface.
    atoms_surfads = attach_adsorbate(
        atoms_surf=atoms_surf,
        atoms_ads=atoms_ads,
        surf_bound=surf_bound,
        site_indices=site_indices,
    )
    # Return the surface+adsorbate atoms object.
    return atoms_surfads

# -------------------------------------------------------------------------------------
# ADSORPTION BIDENTATE
# -------------------------------------------------------------------------------------

def adsorption_bidentate(
    atoms_mol: Atoms,
    atoms_surf: Atoms,
    surf_bound: list,
    site_indices: list,
) -> Atoms:
    """
    Put bi-dentate adsorbate molecule on two surface sites.
    """
    # Copy atoms object.
    atoms_ads = atoms_mol.copy()
    # Get position of the site and bound atom.
    pos_site_atoms = [atoms_surf.positions[indices] for indices in site_indices]
    pos_site = [np.mean(pos, axis=0) for pos in pos_site_atoms]
    pos_bound = atoms_ads.positions[surf_bound]
    # Get adsorbate and site directions.
    d_bound = pos_bound[1] - pos_bound[0]
    d_site = pos_site[1] - pos_site[0]
    l_bound = np.linalg.norm(d_bound)
    l_site = np.linalg.norm(d_site)
    # Set the position of the first bound atom to the origin.
    atoms_ads.translate(-pos_bound[0])
    # Get distance of the two bound atoms from the sites.
    for ii in [0, 1]:
        # Get distance of a bound atom from the site.
        slant = natural_cutoffs(atoms_ads)[surf_bound[ii]]
        slant += np.mean(natural_cutoffs(atoms_surf[site_indices[ii]]))
        if ii == 0:
            # Modify the distance for the first bound atom.
            distance = get_pyramid_height(slant=slant, positions=pos_site_atoms[ii])
        else:
            # Modify the distance for the second bound atom.
            height = get_pyramid_height(slant=slant, positions=pos_site_atoms[ii])
            # Rotate the adsorbate according to the difference in distances.
            a2 = d_bound + [0., 0., height-distance]
            rot_matrix = rotation_matrix(a1=d_bound, a2=a2, b1=[0, 0, 1], b2=[0, 0, 1])
            atoms_ads.positions = np.dot(atoms_ads.positions, rot_matrix.T)
    # Translate adsorbate according to the distance from the site.
    atoms_ads.translate([0., 0., distance])
    # Rotate adsorbate to match the direction of the site.
    rot_matrix = rotation_matrix(a1=d_bound, a2=d_site, b1=[0, 0, 1], b2=[0, 0, 1])
    atoms_ads.positions = np.dot(atoms_ads.positions, rot_matrix.T)
    # Translate adsorbate to the site position.
    atoms_ads.translate(pos_site[0] + (d_site/l_site) * (l_site-l_bound)/2)
    # Add adsorbate to the surface.
    atoms_surfads = attach_adsorbate(
        atoms_surf=atoms_surf,
        atoms_ads=atoms_ads,
        surf_bound=surf_bound,
        site_indices=site_indices,
    )
    # Return the surface+adsorbate atoms object.
    return atoms_surfads

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------