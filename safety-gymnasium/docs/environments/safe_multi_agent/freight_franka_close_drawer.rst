.. _FreightFrankaCloseDrawer-MA:

FreightFrankaCloseDrawer(Multi-Agent)
=====================================


.. list-table::
   :header-rows: 1

   * - Agent
   * - :doc:`../../components_of_environments/agents/freight_franka`

.. image:: ../../_static/images/freight_franka_close_drawer.gif
    :align: center
    :scale: 26 %

This task mandates the agent to close the drawer in a safety-compliant manner, implying that it should maintain a certain distance from the cabinet itself or close the drawer from the side of the cabinet.



Observations
------------



Agent0
^^^^^^

+-----------------+---------------------------------------------------------------------------------------------------+
| Index           | Description                                                                                       |
+=================+===================================================================================================+
| 0 - 2           | Joint DOF values                                                                                  |
+-----------------+---------------------------------------------------------------------------------------------------+
| 3 - 5           | Joint DOF velocities                                                                              |
+-----------------+---------------------------------------------------------------------------------------------------+
| 6 - 7           | Cabinet drawer DOF                                                                                |
+-----------------+---------------------------------------------------------------------------------------------------+
| 8 - 20          | Relative pose between the Franka robot's root and the hand rigid body tensor                      |
+-----------------+---------------------------------------------------------------------------------------------------+
| 21 - 32         | Actions taken by the robot in the joint space                                                     |
+-----------------+---------------------------------------------------------------------------------------------------+
| 33 - 35         | Difference between the xyz pos of freight's root tensor and the handle position                   |
+-----------------+---------------------------------------------------------------------------------------------------+
| 36 - 38         | Difference between the handle position and the hand tip position                                  |
+-----------------+---------------------------------------------------------------------------------------------------+

Agent1
^^^^^^

+-----------------+---------------------------------------------------------------------------------------------------+
| Index           | Description                                                                                       |
+=================+===================================================================================================+
| 0 - 8           | Joint DOF values                                                                                  |
+-----------------+---------------------------------------------------------------------------------------------------+
| 9 - 17          | Joint DOF velocities                                                                              |
+-----------------+---------------------------------------------------------------------------------------------------+
| 18 - 19         | Cabinet drawer DOF                                                                                |
+-----------------+---------------------------------------------------------------------------------------------------+
| 20 - 32         | Relative pose between the Franka robot's root and the hand rigid body tensor                      |
+-----------------+---------------------------------------------------------------------------------------------------+
| 33 - 44         | Actions taken by the robot in the joint space                                                     |
+-----------------+---------------------------------------------------------------------------------------------------+
| 45 - 47         | Difference between the xyz pos of freight's root tensor and the handle position                   |
+-----------------+---------------------------------------------------------------------------------------------------+
| 48 - 50         | Difference between the handle position and the hand tip position                                  |
+-----------------+---------------------------------------------------------------------------------------------------+

Actions
-------

Agent0
^^^^^^

+-----------+----------------------------------------------------------------------------------------------+
| Index     | Description                                                                                  |
+===========+==============================================================================================+
| 0         | x_joint of freight                                                                           |
+-----------+----------------------------------------------------------------------------------------------+
| 1         | y_joint of freight                                                                           |
+-----------+----------------------------------------------------------------------------------------------+
| 2         | z_rotation_joint of freight                                                                  |
+-----------+----------------------------------------------------------------------------------------------+


Agent1
^^^^^^

+-----------+----------------------------------------------------------------------------------------------+
| Index     | Description                                                                                  |
+===========+==============================================================================================+
| 0         | panda_joint1                                                                                 |
+-----------+----------------------------------------------------------------------------------------------+
| 1         | panda_joint2                                                                                 |
+-----------+----------------------------------------------------------------------------------------------+
| 2         | panda_joint3                                                                                 |
+-----------+----------------------------------------------------------------------------------------------+
| 3         | panda_joint4                                                                                 |
+-----------+----------------------------------------------------------------------------------------------+
| 4         | panda_joint5                                                                                 |
+-----------+----------------------------------------------------------------------------------------------+
| 5         | panda_joint6                                                                                 |
+-----------+----------------------------------------------------------------------------------------------+
| 6         | panda_joint7                                                                                 |
+-----------+----------------------------------------------------------------------------------------------+
| 7         | panda_finger_joint1                                                                          |
+-----------+----------------------------------------------------------------------------------------------+
| 8         | panda_finger_joint2                                                                          |
+-----------+----------------------------------------------------------------------------------------------+

Rewards
-------


+------------------------------------------+-----------------------------------+
| State Variable                           | Notation                          |
+==========================================+===================================+
| Hand tip position                        | :math:`p_{hand\_tip}`             |
+------------------------------------------+-----------------------------------+
| Drawer position                          | :math:`p_{drawer}`                |
+------------------------------------------+-----------------------------------+
| Direction of the hand grip               | :math:`\vec{d_{grip}}`            |
+------------------------------------------+-----------------------------------+
| Direction of hand separation             | :math:`\vec{d_{sep}}`             |
+------------------------------------------+-----------------------------------+
| Z-axis direction of the handle           | :math:`\vec{d_{handle\_z}}`       |
+------------------------------------------+-----------------------------------+
| X-axis direction of the handle           | :math:`\vec{d_{handle\_x}}`       |
+------------------------------------------+-----------------------------------+
| Drawer open dof value                    | :math:`d_c`                       |
+------------------------------------------+-----------------------------------+

Distance between the hand tip and the drawer is denoted as:

.. math::
   d = \lVert p_{hand\_tip} - p_{drawer} \rVert_2

**Reward based on this distance**

.. math::
   d_{reward} = \left\{
     \begin{array}{ll}
       2 \times \left(\frac{1}{{1 + d^2}}\right)^2 & \text{if } d \leq 0.1 \\
       \left(\frac{1}{{1 + d^2}}\right)^2 & \text{otherwise}
     \end{array}
   \right.


Orientation match values are:

.. math::
   \omega_{1} = \vec{d_{grip}} \cdot \vec{d_{handle\_z}}

   \omega_{2} = -\vec{d_{sep}} \cdot \vec{d_{handle\_x}}

**Reward for matching the orientation**

.. math::
   r_{rot} = 0.5 \left( \text{sign}(\omega_{1}) \cdot \omega_{1}^2 + \text{sign}(\omega_{2}) \cdot \omega_{2}^2 \right)


**Total Reward**

.. math::
   r = 1.0 \cdot d_{reward} + 0.5 \cdot r_{rot} - 10 \cdot d_c

Costs
-----


+----------------------------------------------+-----------------------------------+
| State Variable                               | Notation                          |
+==============================================+===================================+
| Freight's X-Y Position                       | :math:`f_p`                       |
+----------------------------------------------+-----------------------------------+

Freight positioning cost is based on whether it lies within a defined rectangular zone. This zone is defined by:


+--------------------------------+----------------------------------+
| Axis                           | Range                            |
+================================+==================================+
| X-axis                         | :math:`[-0.25, 0.25]`            |
+--------------------------------+----------------------------------+
| Y-axis                         | :math:`[-0.5, 0.5]`              |
+--------------------------------+----------------------------------+



The cost, :math:`c`, is:

.. math::

    c =
    \begin{cases}
    1 & \text{if } f_p \text{ lies within the zone} \\
    0 & \text{otherwise}
    \end{cases}
