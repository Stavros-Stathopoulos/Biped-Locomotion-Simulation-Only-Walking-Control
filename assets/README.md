# Assets Directory

This directory contains the physical ground truth of the simulation. It strictly houses MuJoCo MJCF XML files, STL/OBJ meshes, and textures.

## Logic & Rules
- **Do not modify the robot's base kinematics.** Keep the original `g1.xml` as close to the manufacturer's spec as possible.
- **Use a Compositional Scene:** `scene.xml` should use the `<include>` tag to bring in `g1.xml`. Add the floor, lighting, and environmental obstacles (like stairs) ONLY in `scene.xml`.
- **Hardware Abstraction:** Any modifications to joint damping, friction, or stiffness belong here, as they represent the physical hardware properties, not software configurations.