
[kafka](https://blog.csdn.net/m0_65931372/article/details/125971395)

[kafka](https://blog.csdn.net/shangmingtao/article/details/79567921?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-79567921-blog-125971395.pc_relevant_multi_platform_whitelistv3&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-79567921-blog-125971395.pc_relevant_multi_platform_whitelistv3&utm_relevant_index=1)

## redis 与 kafka 区别 1
（1）redis的主要作用是用以内存数据库，只是它提供的数据类型list可以用作消息队列而已

而kafka是本身就是消息队列，消息的存储模型只是其中的一个环节，还提供了消息ACK和队列容量、消费速率等消息相关的功能，更加完善

（2）redis 发布订阅除了表示不同的 topic 外，并不支持分组

kafka每个consumer属于一个特定的consumer group（default group）, 同一topic的一条消息只能被同一个consumer group内的一个consumer消费，但多个consumer group可同时消费这一消息。(我的理解是kafka中同一消费组中多个消费者进行消费且同一分组中的只可一个消费者消费同一条消息，避免重复消费且负载均衡更加灵活)

  (3) 处理数据大小的级别不同 

kafka is a distributed, partitiononed,replicated commited logservice. kafka是一个分布式的、易扩展的、安全性高的消息服务系统。kafka提供了类似于JMS的特性，但在设计实现上又完全不同，它并不是基于JMS规范实现的（kafka的实现不包含事务特性性）。kafka对消息的保存时以Topic进行归类的，向Topic发送消息的称谓Producer,从Topic接受消息的称谓Consumer。kafka集群由多个service组成，每个service在kafka集群中被称作broker。kafka集群的作用就是存储从Producer发过来的消息，然后按照一定的规则将消息发送给Consumer。无论是kafka集群本身，还是Producer 或者Consumer，均依赖于zookeeper来管理集群中的信息同步。


## redis 与 kafka 区别 2
redis是一个基于内存的kv数据库，而kafka是分布式发布订阅消息系统.两者本身不是同样一个层次的东西。
redis中有一个queue的数据类型，用来做发布/订阅系统，这个就可以和kafka进行比较了哈。

**存储介质不同**

redis queue数据是存储在内存，虽然有AOF和RDB的持久化方式，但是还是以内存为主。
kafka是存储在硬盘上

**性能不同**

因为存储介质不同，理论上redis queue的性能要优于kafka，但是在实际使用过程，这块体验并不是很明显，通常只有一些高并发场景下需要用redis queue，比如发红包，可以先将红包预先拆解然后push到redis queue，在抢的一瞬间可以很好的支撑并发。

**成本不同**

这边要划重点，划重点，划重点。

kafka存储在硬盘上，成本会比内存小很多，具体差1，2个数量级是有，在数据量非常大的情况下，使用kafka能够节省蛮多服务器成本。最常见的有应用产生的日志，这些日志产生的量级一般都很大，如果有需要进行处理，可以使用kafka队列。

**消息可靠**

redis存储在内存中，一旦服务异常或者宕机，数据就会丢失。相对来说kafka存储在硬盘更加安全。

**订阅机制**

这边也是比较重点，订阅机制主要有两点不同：

kafka消费了之后，可以重新消费。redis消费（lpop）了数据之后，数据就从队列里消失了。kafka里面是偏移量（offset）的概念，可以设置到前面重新消费。
redis只支持单一的消费者，只有topic模式。kafka不光有topic，还支持group分组模式，可以有多个消费组，消费同一个topic的消息。比如应用产生的行为日志，走kafka就很合适，大数据部门可以消费做数据分析，开发部门可以消费做后续的业务逻辑。

**总结**

总结来看，主要就是两点：

存储介质不一样，因为存储介质的不同，造成性能、成本、可靠性的差异。
订阅机制不一样。

**RabbitMQ和kafka的区别**

1.应用场景方面
RabbitMQ：用于实时的，对可靠性要求较高的消息传递上。
kafka：用于处于活跃的流式数据，大数据量的数据处理上。