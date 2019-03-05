se2lam
---
On-SE(2) Localization and Mapping for Ground Vehicles by Fusing Odometry and Vision

### Related Publication

- Fan Zheng, Yun-Hui Liu. "Visual-Odometric Localization and Mapping for Ground Vehicles Using SE(2)-XYZ Constraints". _Proc. IEEE  International Conference on Robotics and Automation (ICRA)_, 2019 \[[pdf](https://fzheng.me/icra/2019.pdf)\]

  [![result in rviz](https://images.gitee.com/uploads/images/2019/0304/152353_36314cbb_874043.jpeg)](https://mycuhk-my.sharepoint.com/:v:/g/personal/1155051778_link_cuhk_edu_hk/EeIO3MJtH5pHsFkIRGHJbLEBRhRBGRRG6pwR19SFCrhQwQ?e=vbSLzS)

### Dependencies

- ROS (tested on Kinetic/Melodic)

- OpenCV 2.4.x / 3.1 above

- g2o ([2016 version](https://github.com/RainerKuemmerle/g2o/releases/tag/20160424_git))

### Build

Build this project as a ROS package

### Demo

1. Download [DatasetRoom.zip](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155051778_link_cuhk_edu_hk/Ef4NuXvLZI1JhfljH9LkNxUB5xrDrCOrRnxwztO5bGKlew?e=U4aind), and extract it. In a terminal, `cd` into `DatasetRoom/`.

   We prepare two packages of odometry measurement data, one is more accurate (`odo_raw_accu.txt`), the other less accurate (`odo_raw_roug.txt`). To use either one of them, copy it to `odo_raw.txt` in `DatasetRoom/`.

2. Download [ORBvoc.bin](http://files.git.oschina.net/group1/M00/00/7A/ZxV3cFeI9XmAby8AAqT02a7stTI369.bin?token=cfa452605a820832918e620075c2ae9f&ts=1551679978&attname=ORBvoc.bin).

3. Run rviz:

   ```
   roscd se2lam
   rosrun rviz rviz -d rviz.rviz
   ```

4. Run se2lam:
   
   ```
   rosrun se2lam test_vn PATH_TO_DatasetRoom PATH_TO_ORBvoc.bin
   ```
   
### Related Project

[izhengfan/se2clam](https://github.com/izhengfan/se2clam)

### License 

[MIT](LICENSE)
