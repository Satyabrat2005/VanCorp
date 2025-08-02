import 'package:flutter/material.dart';
import 'package:animated_text_kit/animated_text_kit.dart';
import 'dart:math';

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
  late final AnimationController _glowController;

  final List<Offset> directions = [
    const Offset(0, -2), // V
    const Offset(-2, 0), // A
    const Offset(2, 0),  // N
    const Offset(0, 2),  // I
    const Offset(1.5, -1.5), // T
    const Offset(-1.5, 1.5), // Y
  ];

  final String text = "VANITY";
  final Color neonGreen = const Color(0xFF1B5801);

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

    _glowController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 3),
    )..repeat(reverse: true);
  }

  @override
  void dispose() {
    for (final controller in _controllers) {
      controller.dispose();
    }
    _glowController.dispose();
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
      body: Stack(
        children: [
          // Dim background
          Container(color: Colors.black),

          // Glowing circular region in top right
          Positioned(
            top: 60,
            right: 30,
            child: AnimatedBuilder(
              animation: _glowController,
              builder: (context, child) {
                double glow = 20 + (_glowController.value * 40);
                return Container(
                  width: 250,
                  height: 250,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    gradient: RadialGradient(
                      colors: [
                        // ignore: deprecated_member_use
                        neonGreen.withOpacity(0.4),
                        // ignore: deprecated_member_use
                        neonGreen.withOpacity(0.05),
                        Colors.transparent,
                      ],
                      radius: 0.9,
                    ),
                    boxShadow: [
                      BoxShadow(
                        // ignore: deprecated_member_use
                        color: neonGreen.withOpacity(0.5),
                        blurRadius: glow,
                        spreadRadius: 10,
                      ),
                    ],
                  ),
                  child: child,
                );
              },
              // Inner content: text and animation
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  AnimatedTextKit(
                    repeatForever: true,
                    animatedTexts: [
                      TypewriterAnimatedText(
                        'Welcome to Vancorp Holdings',
                        textStyle: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                          color: neonGreen,
                        ),
                        speed: const Duration(milliseconds: 100),
                      ),
                    ],
                  ),
                  const SizedBox(height: 16),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: List.generate(text.length, (index) {
                      return SlideTransition(
                        position: _animations[index],
                        child: Text(
                          text[index],
                          style: TextStyle(
                            fontSize: 28,
                            fontWeight: FontWeight.bold,
                            color: neonGreen,
                            letterSpacing: 3,
                          ),
                        ),
                      );
                    }),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
