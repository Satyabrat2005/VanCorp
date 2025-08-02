import 'package:flutter/material.dart';

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Vancorp Holdings',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark(),
      home: const SplashAppleStyle(),
    );
  }
}

class SplashAppleStyle extends StatefulWidget {
  const SplashAppleStyle({super.key});

  @override
  State<SplashAppleStyle> createState() => _SplashAppleStyleState();
}

class _SplashAppleStyleState extends State<SplashAppleStyle> with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<Offset> _circleAnimation;

  @override
  void initState() {
    super.initState();

    _controller = AnimationController(
      duration: const Duration(seconds: 2),
      vsync: this,
    );

    _circleAnimation = Tween<Offset>(
      begin: const Offset(0.5, 1.5), // Start from bottom center
      end: const Offset(-1.0, -1.0), // End at top left corner
    ).animate(CurvedAnimation(parent: _controller, curve: Curves.easeOutBack));

    _controller.forward();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final screenSize = MediaQuery.of(context).size;

    return Scaffold(
      backgroundColor: const Color(0xFF0B0B0D), // Deep blackish gray
      body: Stack(
        children: [
          // Animated Glowing Circle
          SlideTransition(
            position: _circleAnimation,
            child: Align(
              alignment: Alignment.topLeft,
              child: Container(
                margin: const EdgeInsets.only(top: 60, left: 30),
                width: 140,
                height: 140,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  border: Border.all(
                    color: Colors.white.withOpacity(0.6),
                    width: 2.5,
                  ),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.white.withOpacity(0.15),
                      blurRadius: 60,
                      spreadRadius: 8,
                    ),
                  ],
                  gradient: const RadialGradient(
                    colors: [Color(0xFF1F2A3D), Color(0xFF0B0B0D)],
                    center: Alignment.center,
                    radius: 0.85,
                  ),
                ),
                child: const Center(
                  child: Text(
                    "VANITY",
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 18,
                      fontWeight: FontWeight.w600,
                      letterSpacing: 3,
                      fontFamily: 'San Francisco',
                    ),
                  ),
                ),
              ),
            ),
          ),

          // Slogan / Quote below
          Positioned(
            top: screenSize.height * 0.33,
            left: 30,
            right: 30,
            child: Column(
              children: const [
                Text(
                  '"Empowering the shape of ecommerce\nwith Vancorp Holdings"',
                  style: TextStyle(
                    color: Color(0xFFB0B3B8),
                    fontSize: 18,
                    fontWeight: FontWeight.w400,
                    letterSpacing: 1.2,
                    height: 1.5,
                    fontStyle: FontStyle.italic,
                  ),
                  textAlign: TextAlign.left,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
