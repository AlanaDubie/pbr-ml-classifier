import maya.cmds as cmds

class SceneScanner:
    def __init__(self):
        self.objects = []

    def get_selected_meshes(self):
        """Get selected mesh transform objects"""
        selection = cmds.ls(selection=True, long=True, type="transform")

        meshes = []
        for obj in selection:
            shapes = cmds.listRelatives(obj, shapes=True, fullPath=True) or []
            if shapes:
                meshes.append(obj)

        self.objects = meshes
        return meshes

    def get_all_scene_meshes(self):
        """Optional: scan full scene"""
        all_meshes = cmds.ls(type="mesh", long=True)
        transforms = cmds.listRelatives(all_meshes, parent=True, fullPath=True) or []
        self.objects = list(set(transforms))
        return self.objects