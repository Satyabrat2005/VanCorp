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

class MyHomePage extends StatelessWidget {
  const MyHomePage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: const Text('Animated Header'),
        backgroundColor: Colors.transparent,
        elevation: 0,
        centerTitle: true,
      ),
      body: Center(
        child: AnimatedTextKit(
          repeatForever: true,
          animatedTexts: [
            TypewriterAnimatedText(
              'Welcome to your Black App!',
              textStyle: const TextStyle(
                fontSize: 32.0,
                fontWeight: FontWeight.bold,
                color: Colors.white,
              ),
              speed: const Duration(milliseconds: 120),
            ),
          ],
        ),
      ),
    );
  }
}
