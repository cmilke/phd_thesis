import pickle
import numpy

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


def adjust_active_set(active_set, point_list, point_dict, edge_dict):
    active_point, active_edge, active_face = active_set
    if len(edge_dict[active_edge]) <= 1: return active_set

    point_dict[active_point].remove(active_edge)
    available_edges = list(point_dict[active_point])
    for possible_edge in available_edges:
        if len(edge_dict[possible_edge]) > 1:
            point_dict[active_point].remove(possible_edge)
        else:
            active_edge = possible_edge
            active_face = list(edge_dict[active_edge])[0]
            return ( active_point, active_edge, active_face )

    # No edges left for this point;
    # Time to switch to new active_point/active_edge/active_face
    for possible_edge, faces in edge_dict.items():
        if len(faces) == 0:
            print('Wait how did this happen???')
            exit(1)
        elif len(faces) > 1: continue
        for possible_point in possible_edge:
            if possible_point in point_list:
                active_point = possible_point
                active_edge = possible_edge
                active_face = list(faces)[0]
                point_list.remove(active_point)
                return ( active_point, active_edge, active_face )

    # No available edges left
    return None

                


def make_full_3D_render(point_array):
    point_list = list(range(len(point_array)))

    # Create first point
    point_dict = {}
    active_point = 0
    point_list.remove(active_point)
    point_dict[active_point] = set()

    # Create first edge
    edge_dict = {}
    distances = get_distances(point_array, active_point, point_list)
    connecting_point = point_list[ numpy.argmin(distances) ]
    point_dict[connecting_point] = set()
    active_edge = create_edge(point_dict,active_point,connecting_point)  
    edge_dict[active_edge] = set()

    # Create first face
    face_set = set()
    active_list = [ p for p in point_list if p not in active_edge ]
    distance0 = get_distances(point_array, active_edge[0], active_list)
    distance1 = get_distances(point_array, active_edge[1], active_list)
    distance_sum = distance0 + distance1
    closing_point = active_list[ numpy.argmin(distance_sum) ]
    point_dict[closing_point] = set()
    new_edge1 = create_edge(point_dict, active_point, closing_point)
    new_edge2 = create_edge(point_dict, connecting_point, closing_point)
    active_face = create_first_face(active_point, connecting_point, closing_point)
    face_set.add(active_face)
    edge_dict[active_edge].add(active_face)
    edge_dict[new_edge1] = {active_face}
    edge_dict[new_edge2] = {active_face}

    # Begin edge assimilation
    active_set = ( active_point, active_edge, active_face )
    print(active_set)
    print('__________')
    print(point_dict)
    print()
    print(edge_dict)
    print()
    print(face_set)
    print('\n------------\n')

    #while True:
    for i in range(30):
        print(active_set)
        print('__________')
        active_point, active_edge, active_face = active_set
        active_list = [ p for p in point_list if p not in active_face ]
        distance0 = get_distances(point_array, active_edge[0], active_list)
        distance1 = get_distances(point_array, active_edge[1], active_list)
        distance_sum = distance0 + distance1
        closing_point = active_list[ numpy.argmin(distance_sum) ]
        point_dict[closing_point] = set()
        new_edge1 = create_edge(point_dict, active_edge[0], closing_point)
        new_edge2 = create_edge(point_dict, active_edge[1], closing_point)
        print('*p1--->>' + str(active_edge[0]))
        print('*p2--->>' + str(active_edge[1]))
        print('*p3--->>' + str(closing_point))
        print('*1--->>' + str(new_edge1))
        print('*2--->>' + str(new_edge2))
        new_face = create_face(active_edge[0], active_edge[1], closing_point, active_edge)
        print('==--->>' + str(new_face))
        face_set.add(new_face)
        edge_dict[active_edge].add(new_face)
        edge_dict.setdefault(new_edge1, set()).add(new_face)
        edge_dict.setdefault(new_edge2, set()).add(new_face)
        active_set = adjust_active_set(active_set, point_list, point_dict, edge_dict)
        print(point_dict)
        print()
        print(edge_dict)
        print()
        print(face_set)
        print('\n------------\n')
        if active_set is None: break
    print('==============')
    print('|  COMPLETE  |')
    print('==============')
    print(point_dict)
    print()
    print(edge_dict)
    print()
    print(face_set)
    print('\n------------\n')




def main():
    shell_points = pickle.load(open('.shell_points.p','rb'))
    shell_points = shell_points[:10]
    point_array = numpy.array(shell_points)
    make_full_3D_render(point_array)

main()
