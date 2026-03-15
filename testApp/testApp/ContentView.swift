//
//  ContentView.swift
//  testApp
//
//  Created by Sanjib Sarkar on 3/2/26.
//

import SwiftUI

struct ContentView: View {
    @State private var camera = CameraController()

    var body: some View {
        ZStack {
            CameraPreview(session: camera.session)
                .ignoresSafeArea()

            VStack {
                Spacer()
                Text("Live Camera TSET")
                    .padding(8)
                    .background(.ultraThinMaterial)
                    .clipShape(Capsule())
                    .padding(.bottom, 40)
            }
        }
        .onAppear { camera.start() }
        .onDisappear { camera.stop() }
    }
}

//struct ContentView: View {
//    var body: some View {
//        VStack {
//            Image(systemName: "globe")
//                .imageScale(.large)
//                .foregroundStyle(.tint)
//            Text("Hello, world Sanjib!").bold(true)
//        }
//        .padding()
//    }
//}
//
//#Preview {
//    ContentView()
//}
