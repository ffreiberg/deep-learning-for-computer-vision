import tensorflow as tf

# theta1 = tf.Variable([2], dtype=tf.float32)
# theta2 = tf.Variable([1], dtype=tf.float32)
# theta3 = tf.Variable([0], dtype=tf.float32)

theta = tf.Variable( [ 2, 1, 0 ], dtype=tf.float32 )

loss = ( 2 * theta[0] ** 2) + (4 * theta[1]) + tf.maximum(.0, theta[1] + theta[2] )

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run( init )

optimizer = tf.train.GradientDescentOptimizer( .5 )

train = optimizer.minimize( loss )

for i in range(2):
    sess.run( train )
    print( sess.run( theta ) )