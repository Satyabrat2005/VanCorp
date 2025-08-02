import 'package:flutter/material.dart';
import 'package:animated_text_kit/animated_text_kit.dart';

void main() {
  runApp(const MyBlackHomeApp());
}

class MyBlackHomeApp extends StatelessWidget {
  const MyBlackHomeApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Vancorp Holdings',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark(),
      home: const MyHomePage(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key});

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> with TickerProviderStateMixin {
  late final List<AnimationController> _controllers;
  late final List<Animation<Offset>> _animations;

  final List<Offset> directions = [
    const Offset(0, -2), // V - from top
    const Offset(-2, 0), // A - from left
    const Offset(2, 0),  // N - from right
    const Offset(0, 2),  // I - from bottom
    const Offset(1.5, -1.5), // T - from top right
    const Offset(-1.5, 1.5), // Y - from bottom left
  ];

  final String text = "VANITY";

  @override
  void initState() {
    super.initState();

    _controllers = List.generate(
      text.length,
      (index) => AnimationController(
        vsync: this,
        duration: const Duration(milliseconds: 700),
      ),
    );

    _animations = List.generate(
      text.length,
      (index) => Tween<Offset>(
        begin: directions[index],
        end: Offset.zero,
      ).animate(CurvedAnimation(
        parent: _controllers[index],
        curve: Curves.easeOut,
      )),
    );

    for (int i = 0; i < _controllers.length; i++) {
      Future.delayed(Duration(milliseconds: i * 250), () {
        _controllers[i].forward();
      });
    }
  }

  @override
  void dispose() {
    for (final controller in _controllers) {
      controller.dispose();
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: const Text('Vancorp Holdings'),
        backgroundColor: Colors.transparent,
        elevation: 0,
        centerTitle: true,
        leading: const Icon(Icons.phone),
      ),
      body: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          AnimatedTextKit(
            repeatForever: true,
            animatedTexts: [
              TypewriterAnimatedText(
                'Welcome to Vancorp Holdings',
                textStyle: const TextStyle(
                  fontSize: 28.0,
                  fontWeight: FontWeight.bold,
                  color: Colors.white,
                ),
                speed: const Duration(milliseconds: 100),
              ),
            ],
          ),
          const SizedBox(height: 40),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: List.generate(text.length, (index) {
              return SlideTransition(
                position: _animations[index],
                child: Text(
                  text[index],
                  style: const TextStyle(
                    fontSize: 40,
                    fontWeight: FontWeight.bold,
                    color: Colors.white,
                  ),
                ),
              );
            }),
          ),
        ],
      ),
    );
  }
}
