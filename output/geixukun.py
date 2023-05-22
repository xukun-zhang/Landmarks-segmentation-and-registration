import openmesh as om
import trimesh
import math
def qici2rot(qici):
    '''
    采用openmesh加载.ply时，做mesh的旋转、平移变换用的
    4*4齐次变为3*3旋转矩阵
    '''
    s = qici.shape
    if s[0]!=s[1]:
        raise ValueError('smy: Matrix must be a 4*4 qici ndarray',s)
    n = s[0]
    rot = qici[0:n-1,0:n-1]
    return rot

def load_obj(obj_file, trans=[0,0,0]):
    # -----------openmesh版本-------------------#
    # load_obj用openmesh重写，因为mesh是用openmesh写的，用trimesh读取时，总会出现拓扑结构问题。
    mesh = om.read_trimesh(obj_file)
    verts = mesh.points()
    faces = mesh.face_vertex_indices()

    # 我输入的是轴角式表示法, 输出4*4齐次形式的旋转矩阵
    R = trimesh.transformations.rotation_matrix(math.radians(90), [1, 0, 0])
    R = qici2rot(R)
    # 实际上, 如果有3*3的旋转矩阵, 直接在这里乘就可以了,不需要前面的
    verts = verts.dot(R)
    # 为了保证用numpy算的结果与原来trimesh的一致，做一点变换
    # numpy矩阵乘法的结果与trimesh库的乘法坐标系有点差异,好像一个是左手,一个是右手,
    # 具体要不要执行下面这个, 可以用meshlab看一下;
    verts[:, 1] = -verts[:, 1]
    verts[:, 2] = -verts[:, 2]

    # 平移就直接加就行了
    T = trans
    verts += T
    mesh_new = om.TriMesh()
    mesh_new.add_vertices(verts)
    mesh_new.add_faces(faces)
    om.write_mesh('./output/mesh_new.obj',mesh_new)
    # return mesh, verts, faces

if __name__ == '__main__':
    load_obj(obj_file='123.obj', trans=[0,0,0])