from setuptools import find_packages, setup

package_name = 'turtlebot_dqn'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    # data_files=[
    #     ('share/ament_index/resource_index/packages',
    #         ['resource/' + package_name]),
    #     ('share/' + package_name, ['package.xml']),
    # ],
    data_files=[
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    ('share/' + package_name + '/launch', ['launch/dqn_test.launch.py']),
],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='asad',
    maintainer_email='asad@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': ['dqn_node = turtlebot_dqn.dqn_node:main',
                            'bumper_reset_node = turtlebot_dqn.bumper_reset_node:main',
        ],
    },
)
