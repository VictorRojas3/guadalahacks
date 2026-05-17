//
//  Item.swift
//  Calli
//
//  Created by Victor Alejandro  Rojas Gamez  on 16/05/26.
//

import Foundation
import SwiftData

@Model
final class Item {
    var timestamp: Date
    
    init(timestamp: Date) {
        self.timestamp = timestamp
    }
}
