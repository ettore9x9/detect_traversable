from setuptools import setup

package_name = 'detect_traversable'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ettore',
    maintainer_email='ettoresani0@gmail.com',
    description='A node to detect the traversable area using Deep Learning',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detect_traversable_sub = detect_traversable.DetectTraversable:main',
        ],
    },
)
