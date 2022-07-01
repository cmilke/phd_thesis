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


def make_full_3D_render(point_array):
    import alphashape
    geometry = alphashape.alphashape(point_array, 1.2)
    convert_to_ply_mesh(geometry.vertices, geometry.faces)


def main():
    raw_shell_points = pickle.load(open('.shell_points.p','rb'))
    shell_points = [ [x/2,y/10,z] for x,y,z in raw_shell_points ]
    point_array = numpy.array(shell_points)
    print(len(point_array))
    make_full_3D_render(point_array)

main()
