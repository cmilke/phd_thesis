import pickle
import numpy


def convert_to_ply_mesh(point_array, face_set):
    # ply header
    ply_output  = 'ply\n'
    ply_output += 'format ascii 1.0\n'
    ply_output += f'element vertex {len(point_array)}\n'
    ply_output += 'property float x\n'
    ply_output += 'property float y\n'
    ply_output += 'property float z\n'
    ply_output += f'element face {len(face_set)}\n'
    ply_output += 'property list uchar int vertex_index\n'  # Number of vertices for each face
    #element edge 5                        { five edges in object }
    #property int vertex1                  { index to first vertex of edge }
    #property int vertex2                  { index to second vertex }
    ply_output += 'end_header\n'

    # ply vertex list
    for x,y,z in point_array:
        ply_output += f'{x} {y} {z}\n'

    # ply face list
    for a,b,c in face_set:
        ply_output += f'3 {a} {b} {c}\n'

    f = open("limit_mesh.ply", "w")
    f.write(ply_output)
    f.close()


def get_distances(point_array, reference, point_list):
    reference_point = point_array[reference]
    other_points = point_array[point_list]
    distances = numpy.linalg.norm(reference_point-other_points, axis=1)
    return distances

def create_face(p1,p2,p3,shared_edge): #TODO permute this correctly
    return tuple( sorted([p1,p2,p3]) )

def create_first_face(p1,p2,p3): #TODO orient to align with outwards vector
    return tuple( sorted([p1,p2,p3]) )

def create_edge(point_dict,p1,p2):
    new_edge = tuple( sorted([p1,p2]) )
    point_dict[p1].add(new_edge)
    point_dict[p2].add(new_edge)
    return new_edge


def adjust_active_set(active_set, point_dict, edge_dict):
    active_point, active_edge, active_list = active_set
    if len(active_list) == 0: return None
    point_ranks = sorted([ (len(s),p) for p,s in point_dict.items() ])
    
    for edge_count, active_point in point_ranks:
        available_edges = list(point_dict[active_point])
        for possible_edge in available_edges:
            if len(edge_dict[possible_edge]) == 1:
                active_edge = possible_edge
                return ( active_point, active_edge, active_list )

    # No available edges left
    return None

                


def make_full_3D_render(point_array):
    point_list = list(range(len(point_array)))

    # Create first point
    point_dict = {}
    active_list = point_list.copy()
    active_point = 0
    active_list.remove(active_point)
    point_dict[active_point] = set()

    # Create first edge
    edge_dict = {}
    distances = get_distances(point_array, active_point, active_list)
    connecting_point = active_list[ numpy.argmin(distances) ]
    active_list.remove(connecting_point)
    point_dict[connecting_point] = set()
    active_edge = create_edge(point_dict,active_point,connecting_point)  
    edge_dict[active_edge] = set()

    # Create first face
    face_set = set()
    distance0 = get_distances(point_array, active_edge[0], active_list)
    distance1 = get_distances(point_array, active_edge[1], active_list)
    distance_sum = distance0 + distance1
    closing_point = active_list[ numpy.argmin(distance_sum) ]
    active_list.remove(closing_point)
    point_dict[closing_point] = set()
    new_edge1 = create_edge(point_dict, active_point, closing_point)
    new_edge2 = create_edge(point_dict, connecting_point, closing_point)
    active_face = create_first_face(active_point, connecting_point, closing_point)
    face_set.add(active_face)
    edge_dict[active_edge].add(active_face)
    edge_dict[new_edge1] = {active_face}
    edge_dict[new_edge2] = {active_face}

    # Begin edge assimilation
    active_set = ( active_point, active_edge, active_list )
    #print(active_set)
    #print('__________')
    #print(point_dict)
    #print()
    #print(edge_dict)
    #print()
    #print(face_set)
    #print('\n------------\n')

    while True:
    #for i in range(1000):
        #print(i)
        #print(active_set)
        #print('__________')
        print(len(active_list))
        active_point, active_edge, active_list = active_set
        distance0 = get_distances(point_array, active_edge[0], active_list)
        distance1 = get_distances(point_array, active_edge[1], active_list)
        distance_sum = distance0 + distance1
        closing_point = active_list[ numpy.argmin(distance_sum) ]
        active_list.remove(closing_point)
        point_dict[closing_point] = set()
        new_edge1 = create_edge(point_dict, active_edge[0], closing_point)
        new_edge2 = create_edge(point_dict, active_edge[1], closing_point)
        #print('*p1--->>' + str(active_edge[0]))
        #print('*p2--->>' + str(active_edge[1]))
        #print('*p3--->>' + str(closing_point))
        #print('*1--->>' + str(new_edge1))
        #print('*2--->>' + str(new_edge2))
        new_face = create_face(active_edge[0], active_edge[1], closing_point, active_edge)
        #print('==--->>' + str(new_face))
        face_set.add(new_face)
        edge_dict[active_edge].add(new_face)
        edge_dict.setdefault(new_edge1, set()).add(new_face)
        edge_dict.setdefault(new_edge2, set()).add(new_face)
        active_set = adjust_active_set(active_set, point_dict, edge_dict)
        #print(point_dict)
        #print()
        #print(edge_dict)
        #print()
        #print(face_set)
        #print('\n------------\n')
        if active_set is None: break
    #print('==============')
    #print('|  COMPLETE  |')
    #print('==============')
    #print(point_dict)
    #print()
    #print(edge_dict)
    #print()
    #print(face_set)
    #print('\n------------\n')

    convert_to_ply_mesh(point_array, face_set)




def main():
    shell_points = pickle.load(open('.shell_points.p','rb'))
    #shell_points = shell_points[:1000]
    point_array = numpy.array(shell_points)
    make_full_3D_render(point_array)

main()
