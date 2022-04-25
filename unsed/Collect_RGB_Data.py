#!/usr/bin/env python3


import glob
import os
import sys
import time

try:
    sys.path.append(glob.glob('${CARLA_ROOT}/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random
import time 

def main():
    actor_list = []

    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)

        world = client.get_world()

        blueprint_library = world.get_blueprint_library()

        bp = random.choice(blueprint_library.filter('vehicle'))

        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)

        transform = random.choice(world.get_map().get_spawn_points())

        vehicle = world.spawn_actor(bp, transform)

        actor_list.append(vehicle)
        print('created %s' % vehicle.type_id)

        vehicle.set_autopilot(True)

        world.tick()

        #get spectator 

        spectator = world.get_spectator()
        world_snapshot = world.wait_for_tick()
        actor_snapshot =  world_snapshot.find(vehicle.id)
        spectator.set_transform(actor_snapshot.get_transform())

        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute("image_size_x",str(128))
        camera_bp.set_attribute("image_size_y",str(128))
        camera_bp.set_attribute("fov",str(105))
        cam_location = carla.Location(2,1,2)
        cam_rotation = carla.Rotation(0,0,0)
        camera_transform = carla.Transform(cam_location, cam_rotation)
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle, 
                                    attachment_type=carla.AttachmentType.Rigid)
        actor_list.append(camera)
        print('created %s' % camera.type_id)


        # Now we register the function that will be called each time the sensor
        # receives an image. In this example we are saving the image to disk
        # converting the pixels to gray-scale.
        camera.listen(lambda image: image.save_to_disk('image_collection/%06d.png' % image.frame))

     
    
        time.sleep(300)

    finally:

        print('destroying actors')
        camera.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print('done.')


if __name__ == '__main__':

    main()
